"""工具增强模块。

负责构造复杂度工具结果和知识图谱工具结果。"""


import re

from .db_source import fetch_relations_for_nodes, search_knowledge_nodes
from .deepseek import extract_complexities


def _kw(text: str) -> list[str]:
    """抽取问题里的关键词，同时补充短 n-gram 提高命中率。"""
    words = re.findall(r'[\w\u4e00-\u9fa5]+', (text or '').lower())
    out: list[str] = []
    for w in words:
        if len(w) >= 2 and w not in out:
            out.append(w)
        # For Chinese phrases, add short n-grams to increase DB hit rate.
        if re.search(r'[\u4e00-\u9fa5]', w) and len(w) >= 4:
            for n in (2, 3, 4):
                for i in range(0, len(w) - n + 1):
                    g = w[i:i + n]
                    if g not in out:
                        out.append(g)
    return out


def build_complexity_tool_chunk(refs: list[dict], algorithm_id: int | None = None) -> dict | None:
    """从检索结果里抽取复杂度信息，构造工具结果 chunk。"""
    merged = '\n'.join((r.get('content') or '') for r in refs[:8])
    complexity_hits = extract_complexities(merged)
    if not complexity_hits:
        return None
    return {
        'id': -1,
        'algorithm_id': algorithm_id or 0,
        'chunk_type': 'tool',
        'title': '复杂度工具结果',
        'content': '从检索片段中提取到复杂度表示：' + '、'.join(complexity_hits),
        'source': 'tool_complexity',
        'score': 1.0,
    }


def build_kg_tool_chunk(question: str, algorithm_id: int | None = None) -> dict | None:
    """根据问题查询知识图谱，并构造工具结果 chunk。"""
    nodes = []
    for token in _kw(question)[:4]:
        rows = search_knowledge_nodes(token, limit=3)
        for r in rows:
            if not any(x['id'] == r['id'] for x in nodes):
                nodes.append(r)
        if len(nodes) >= 6:
            break

    if not nodes:
        for r in search_knowledge_nodes('数据结构', limit=6):
            if not any(x['id'] == r['id'] for x in nodes):
                nodes.append(r)

    if not nodes:
        return None

    node_ids = [int(x['id']) for x in nodes]
    rels = fetch_relations_for_nodes(node_ids, limit=12)

    lines = ['相关知识点：']
    for n in nodes[:6]:
        desc = (n.get('description') or '').strip()
        lines.append(f"- [{n['id']}] {n['name']}（{n.get('type') or '未知类型'}）{('：' + desc) if desc else ''}")

    if rels:
        lines.append('相关关系：')
        for r in rels[:10]:
            lines.append(f"- {r.get('source_name', '')} --{r.get('relation_type', '')}--> {r.get('target_name', '')}")

    return {
        'id': -2,
        'algorithm_id': algorithm_id or 0,
        'chunk_type': 'tool',
        'title': '知识图谱工具结果',
        'content': '\n'.join(lines),
        'source': 'tool_kg',
        'score': 1.0,
    }


def build_tool_chunks(question: str, refs: list[dict], algorithm_id: int | None = None) -> list[dict]:
    """统一收集当前问题可用的所有工具结果。"""
    tool_chunks = []

    complexity_chunk = build_complexity_tool_chunk(refs, algorithm_id=algorithm_id)
    if complexity_chunk:
        tool_chunks.append(complexity_chunk)

    kg_chunk = build_kg_tool_chunk(question, algorithm_id=algorithm_id)
    if kg_chunk:
        tool_chunks.append(kg_chunk)

    return tool_chunks
