"""大模型能力封装层。

把 DeepSeek 的调用统一封装成问答、问题扩展、三元组抽取、问答合成等能力。"""


import json
import re

import requests

from .config import settings


def deepseek_chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1400) -> str:
    """最底层的模型调用接口，其他能力最终都通过它访问大模型。"""
    if not settings.deepseek_api_key:
        raise RuntimeError('DEEPSEEK_API_KEY 未配置')
    payload = {
        'model': settings.deepseek_model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    headers = {
        'Authorization': f'Bearer {settings.deepseek_api_key}',
        'Content-Type': 'application/json',
    }
    with requests.Session() as sess:
        # Avoid inheriting broken local proxy settings.
        sess.trust_env = False
        resp = sess.post(settings.deepseek_base_url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']


def _extract_json_array(raw: str) -> list:
    """从模型输出中尽量提取 JSON 数组。"""
    if not raw:
        return []
    st = raw.find('[')
    ed = raw.rfind(']')
    if st == -1 or ed == -1 or ed <= st:
        return []
    try:
        arr = json.loads(raw[st:ed + 1])
    except Exception:
        return []
    return arr if isinstance(arr, list) else []


def expand_questions(question: str, max_n: int = 3) -> list[str]:
    """把用户问题扩展成多个更利于检索的候选问法。"""
    q = (question or '').strip()
    if not q:
        return []

    prompt = f"""
请把用户问题改写成更利于检索的候选问法，返回JSON数组（仅数组）。
要求：
1) 保留原意；
2) 不引入外部事实；
3) 输出{max_n}条以内短句。

用户问题：{q}
"""
    try:
        raw = deepseek_chat(
            [
                {'role': 'system', 'content': '你是检索优化助手，只输出JSON数组。'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        arr = _extract_json_array(raw)
        out: list[str] = []
        for x in arr:
            s = str(x).strip()
            if s and s != q:
                out.append(s)
        return out[:max_n]
    except Exception:
        return []


def extract_triples(answer: str, max_n: int = 6) -> list[dict]:
    """从回答中抽取实体-关系-实体三元组，用于记忆更新。"""
    text = (answer or '').strip()
    if not text:
        return []

    prompt = f"""
请从下面回答中抽取实体关系三元组，输出JSON数组，元素格式：
{{"head":"实体1","relation":"关系","tail":"实体2"}}
要求：
1) 仅根据输入文本，不编造；
2) 关系尽量简短准确；
3) 最多{max_n}条。

文本：{text[:3800]}
"""
    try:
        raw = deepseek_chat(
            [
                {'role': 'system', 'content': '你是信息抽取助手，只输出JSON数组。'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.1,
            max_tokens=700,
        )
        arr = _extract_json_array(raw)
        out = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            head = str(item.get('head', '')).strip()
            relation = str(item.get('relation', '')).strip()
            tail = str(item.get('tail', '')).strip()
            if head and relation and tail:
                out.append({'head': head, 'relation': relation, 'tail': tail})
        return out[:max_n]
    except Exception:
        return []


def synthesize_qa(algorithm_name: str, step_text: str, code: str) -> list[dict]:
    """基于步骤和代码合成基础问答对。"""
    prompt = f"""
请基于给定算法步骤和代码生成3组问答，返回JSON数组，元素格式：{{"question":"...","answer":"..."}}。
要求：只基于输入内容，不编造外部知识；覆盖原理、复杂度/边界、实现细节。

算法名：{algorithm_name}
步骤：{step_text[:3000]}
代码：{code[:6000]}
"""
    raw = deepseek_chat(
        [
            {'role': 'system', 'content': '你是算法助教，输出严格JSON数组。'},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    arr = _extract_json_array(raw)
    out = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        q = str(x.get('question', '')).strip()
        a = str(x.get('answer', '')).strip()
        if q and a:
            out.append({'question': q, 'answer': a})
    return out[:3]


def extract_complexities(text: str) -> list[str]:
    """从文本中提取 O(...) 形式的复杂度表达。"""
    text = text or ''
    hits = re.findall(r'O\([^\)]+\)', text)
    uniq = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    return uniq[:8]
