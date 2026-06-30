"""LLM-as-Judge 评测模块。

用 DeepSeek 对问答系统的答案质量进行打分，
适用于长文本生成场景（ROUGE/BLEU 不适合时的替代方案）。

评分维度（各 1-5 分）：
  - accuracy:    准确性（答案是否正确，无错误知识）
  - completeness: 完整性（是否覆盖了核心知识点）
  - clarity:     清晰度（是否条理清晰、易于理解）
  - relevance:   相关性（是否紧扣问题，没有跑题）

综合分 = 四维度均值
"""

import json
import re
from typing import Optional

from .deepseek import deepseek_chat


# ------------------------------------------------------------------ #
# 单条评分
# ------------------------------------------------------------------ #

_JUDGE_SYSTEM = (
    "你是一位严格、公正的算法教学质量评审专家。"
    "你的任务是根据给定的评分标准，对学生问答系统的回答进行客观评分。"
    "只输出 JSON 对象，不要输出任何其他内容。"
)

_JUDGE_PROMPT = """请对以下问答进行评分。

【问题】
{question}

【参考答案】
{reference}

【系统回答】
{prediction}

【评分标准】
请从以下四个维度各给出 1-5 的整数分（1=很差，3=一般，5=很好）：
- accuracy（准确性）：回答中的知识点是否正确，有无明显错误
- completeness（完整性）：是否覆盖了参考答案中的核心知识点
- clarity（清晰度）：表达是否清晰、有条理，易于学生理解
- relevance（相关性）：是否紧扣问题，没有大量无关内容

【输出格式】（只输出这个 JSON，不要有其他文字）
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "relevance": <1-5>,
  "reason": "<一句话说明主要优缺点>"
}}
"""


def judge_single(
    question: str,
    prediction: str,
    reference: str,
) -> dict:
    """用 LLM 对单条答案打分，返回各维度分数和综合分。"""
    if not prediction or not prediction.strip():
        return {
            "accuracy": 1, "completeness": 1,
            "clarity": 1, "relevance": 1,
            "overall": 1.0, "reason": "答案为空",
            "raw": "",
        }

    prompt = _JUDGE_PROMPT.format(
        question=question[:500],
        reference=reference[:800],
        prediction=prediction[:1200],
    )

    try:
        raw = deepseek_chat(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as e:
        return {
            "accuracy": 0, "completeness": 0,
            "clarity": 0, "relevance": 0,
            "overall": 0.0, "reason": f"API调用失败: {e}",
            "raw": "",
        }

    # 提取 JSON
    obj = _extract_json(raw)
    if obj is None:
        return {
            "accuracy": 0, "completeness": 0,
            "clarity": 0, "relevance": 0,
            "overall": 0.0, "reason": "解析失败",
            "raw": raw,
        }

    acc  = _clamp(obj.get("accuracy", 0))
    comp = _clamp(obj.get("completeness", 0))
    clar = _clamp(obj.get("clarity", 0))
    relv = _clamp(obj.get("relevance", 0))
    overall = round((acc + comp + clar + relv) / 4, 3)

    return {
        "accuracy":     acc,
        "completeness": comp,
        "clarity":      clar,
        "relevance":    relv,
        "overall":      overall,
        "reason":       str(obj.get("reason", "")),
        "raw":          raw,
    }


# ------------------------------------------------------------------ #
# 批量评分
# ------------------------------------------------------------------ #

def judge_batch(
    questions: list[str],
    predictions: list[str],
    references: list[str],
    verbose: bool = True,
) -> dict:
    """批量评分，返回每条详情和平均分。"""
    assert len(questions) == len(predictions) == len(references), \
        "questions / predictions / references 长度必须一致"

    details = []
    total = len(questions)

    for i, (q, p, r) in enumerate(zip(questions, predictions, references), 1):
        if verbose:
            print(f"\r  LLM评分进度: {i}/{total}", end="", flush=True)
        score = judge_single(q, p, r)
        score["index"] = i
        details.append(score)

    if verbose:
        print()

    # 过滤掉调用失败的（overall=0）
    valid = [d for d in details if d["overall"] > 0]
    n = len(valid) or 1

    avg = {
        "accuracy":     round(sum(d["accuracy"]     for d in valid) / n, 3),
        "completeness": round(sum(d["completeness"] for d in valid) / n, 3),
        "clarity":      round(sum(d["clarity"]      for d in valid) / n, 3),
        "relevance":    round(sum(d["relevance"]    for d in valid) / n, 3),
        "overall":      round(sum(d["overall"]      for d in valid) / n, 3),
    }

    return {
        "average": avg,
        "details": details,
        "valid_count": len(valid),
        "total_count": total,
    }


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #

def _extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取第一个 JSON 对象。"""
    if not text:
        return None
    st = text.find("{")
    ed = text.rfind("}")
    if st == -1 or ed == -1 or ed <= st:
        return None
    try:
        return json.loads(text[st: ed + 1])
    except Exception:
        return None


def _clamp(val) -> int:
    """把分数限制在 1-5 的整数范围内。"""
    try:
        v = int(float(val))
        return max(1, min(5, v))
    except Exception:
        return 1
