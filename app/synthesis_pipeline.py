"""数据合成流水线。

把标题、上下文、大纲、多轮对话和最终答案串成一套合成流程。"""


from __future__ import annotations

import json
from typing import Any

from .db_source import fetch_algorithms
from .deepseek import deepseek_chat
from .rag import retrieve
from .store import insert_chunks


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 对象。"""
    if not raw:
        return None
    st = raw.find('{')
    ed = raw.rfind('}')
    if st == -1 or ed == -1 or ed <= st:
        return None
    try:
        obj = json.loads(raw[st : ed + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _context_from_row(row: dict, top_k: int = 6) -> str:
    """围绕一个算法标题检索相关资料，作为合成上下文。"""
    aids = int(row['algorithm_id'])
    title = row['algorithm_name']
    refs = retrieve(title, top_k=top_k, algorithm_id=aids, mode='hybrid', use_expansion=True, include_memory=False)
    parts = []
    for ref in refs[:top_k]:
        parts.append(f"[{ref.get('chunk_type')}#{ref.get('id')}]\n{ref.get('content')}")
    return '\n\n'.join(parts)


def _make_outline(title: str, context: str) -> list[str]:
    """先根据标题和资料生成初始回答大纲。"""
    prompt = f"""
请基于给定标题和资料，为教学问答生成一个回答大纲，只输出JSON对象：
{{"outline":["要点1","要点2","要点3"]}}

标题：{title}
资料：
{context[:5000]}
"""
    raw = deepseek_chat(
        [
            {'role': 'system', 'content': '你是教学内容规划器，只输出JSON对象。'},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    obj = _extract_json_object(raw) or {}
    outline = obj.get('outline')
    if isinstance(outline, list):
        out = [str(x).strip() for x in outline if str(x).strip()]
        if out:
            return out[:8]
    return [f'{title}的定义与核心思想', f'{title}的实现步骤', f'{title}的典型应用与注意点']


def _make_dialogue(title: str, context: str, outline: list[str]) -> list[dict]:
    """模拟多视角学生与专家之间的教学对话。"""
    prompt = f"""
请围绕给定标题、资料和回答大纲，生成多视角教学对话，只输出JSON对象：
{{
  "dialogue":[
    {{"role":"student_basic","question":"..." }},
    {{"role":"expert","answer":"..." }},
    {{"role":"student_exam","question":"..." }},
    {{"role":"expert","answer":"..." }}
  ]
}}

要求：
1. 覆盖基础理解、实现细节、复杂度/边界、考试应用等不同视角；
2. 问题和回答都要基于资料；
3. 对话轮数控制在8到12条。

标题：{title}
大纲：{json.dumps(outline, ensure_ascii=False)}
资料：
{context[:6000]}
"""
    raw = deepseek_chat(
        [
            {'role': 'system', 'content': '你是教学数据合成器，只输出JSON对象。'},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.3,
        max_tokens=1600,
    )
    obj = _extract_json_object(raw) or {}
    arr = obj.get('dialogue')
    out = []
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            role = str(item.get('role', '')).strip()
            question = str(item.get('question', '')).strip()
            answer = str(item.get('answer', '')).strip()
            if role and (question or answer):
                out.append({'role': role, 'question': question, 'answer': answer})
    return out[:12]


def _refine_outline(title: str, outline: list[str], dialogue: list[dict]) -> list[str]:
    """根据对话历史把初始大纲进一步细化。"""
    prompt = f"""
请基于初始大纲和对话历史，生成更细化的最终回答大纲，只输出JSON对象：
{{"outline":["章节1","章节2","章节3"]}}

标题：{title}
初始大纲：{json.dumps(outline, ensure_ascii=False)}
对话历史：{json.dumps(dialogue, ensure_ascii=False)[:6000]}
"""
    raw = deepseek_chat(
        [
            {'role': 'system', 'content': '你是教学大纲优化器，只输出JSON对象。'},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    obj = _extract_json_object(raw) or {}
    arr = obj.get('outline')
    if isinstance(arr, list):
        out = [str(x).strip() for x in arr if str(x).strip()]
        if out:
            return out[:10]
    return outline


def _final_answer(title: str, refined_outline: list[str], dialogue: list[dict], context: str) -> str:
    """综合细化大纲、对话和资料生成最终答案。"""
    prompt = f"""
请基于标题、细化大纲、对话历史和资料，生成最终教学答案。
要求：
1. 层次清晰；
2. 适合教学场景；
3. 尽量覆盖原理、步骤、复杂度、实现细节和易错点；
4. 不编造资料中没有的结论。

标题：{title}
细化大纲：{json.dumps(refined_outline, ensure_ascii=False)}
对话历史：{json.dumps(dialogue, ensure_ascii=False)[:7000]}
资料：
{context[:7000]}
"""
    return deepseek_chat(
        [
            {'role': 'system', 'content': '你是算法教学专家。'},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.2,
        max_tokens=1800,
    )


def synthesize_dialogue_qa(limit: int = 5, algorithm_id: int | None = None) -> dict[str, Any]:
    """执行整条数据合成流水线，并把结果写回知识库。"""
    rows = fetch_algorithms(limit=limit if algorithm_id is None else None)
    if algorithm_id is not None:
        rows = [row for row in rows if int(row['algorithm_id']) == int(algorithm_id)]

    generated = []
    to_insert = []
    for row in rows[: limit if algorithm_id is None else len(rows)]:
        title = row['algorithm_name']
        aid = int(row['algorithm_id'])
        context = _context_from_row(row)
        if not context:
            context = '\n'.join(
                [
                    row.get('step_text') or '',
                    row.get('analysis_text') or '',
                    row.get('code') or '',
                ]
            )[:6000]

        outline = _make_outline(title, context)
        dialogue = _make_dialogue(title, context, outline)
        refined_outline = _refine_outline(title, outline, dialogue)
        final_answer = _final_answer(title, refined_outline, dialogue, context)

        primary_question = f'{title}应该如何理解和掌握？'
        metadata = {
            'kind': 'synth_dialog_qa',
            'algorithm_name': title,
            'outline': outline,
            'refined_outline': refined_outline,
            'dialogue': dialogue,
            'primary_question': primary_question,
        }
        to_insert.append(
            {
                'algorithm_id': aid,
                'chunk_type': 'qa',
                'title': f'{title}-多视角合成问答',
                'content': f"Q: {primary_question}\nA: {final_answer}",
                'source': 'deepseek_dialogue_synth',
                'metadata_json': json.dumps(metadata, ensure_ascii=False),
            }
        )
        generated.append(
            {
                'algorithm_id': aid,
                'algorithm_name': title,
                'question': primary_question,
                'outline': outline,
                'refined_outline': refined_outline,
                'dialogue_turns': len(dialogue),
            }
        )

    inserted = insert_chunks(to_insert)
    return {
        'generated_count': len(generated),
        'inserted_chunks': inserted,
        'items': generated,
    }
