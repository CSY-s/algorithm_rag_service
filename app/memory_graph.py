"""结构化记忆图模块。

把 memory chunk 解析成三元组，并按问题选出一个小型记忆子图。"""


import json
import re
from collections import defaultdict
from typing import Any

from .store import get_chunks


def _tokens(text: str) -> set[str]:
    """把文本切成简单 token 集合，用于做匹配。"""
    return {x for x in re.findall(r'[\w\u4e00-\u9fa5]+', (text or '').lower()) if len(x) >= 2}


def _parse_memory_chunk(row: dict) -> list[dict]:
    """从 memory chunk 的 metadata 中解析三元组。"""
    meta_raw = row.get('metadata_json') or ''
    triples = []
    try:
        meta = json.loads(meta_raw)
        arr = meta.get('triples') if isinstance(meta, dict) else None
        if isinstance(arr, list):
            for item in arr:
                if not isinstance(item, dict):
                    continue
                head = str(item.get('head', '')).strip()
                relation = str(item.get('relation', '')).strip()
                tail = str(item.get('tail', '')).strip()
                if head and relation and tail:
                    triples.append(
                        {
                            'head': head,
                            'relation': relation,
                            'tail': tail,
                            'chunk_id': row.get('id'),
                            'algorithm_id': row.get('algorithm_id'),
                            'source': row.get('source'),
                        }
                    )
    except Exception:
        return []
    return triples


def load_memory_triples(algorithm_id: int | None = None) -> list[dict]:
    """加载指定算法范围内的所有记忆三元组。"""
    rows = get_chunks(algorithm_id=algorithm_id, include_memory=True)
    out = []
    for row in rows:
        if row.get('chunk_type') != 'memory':
            continue
        out.extend(_parse_memory_chunk(row))
    return out


def select_memory_subgraph(question: str, algorithm_id: int | None = None, top_k: int = 6) -> dict[str, Any]:
    """根据问题选出最相关的记忆三元组及其邻居。"""
    triples = load_memory_triples(algorithm_id=algorithm_id)
    if not triples:
        return {'selected': [], 'neighbors': [], 'yaml': ''}

    q_tokens = _tokens(question)
    scored = []
    for triple in triples:
        triple_text = f"{triple['head']} {triple['relation']} {triple['tail']}"
        overlap = len(q_tokens & _tokens(triple_text))
        if overlap <= 0:
            continue
        score = overlap
        if any(tok in triple['head'].lower() for tok in q_tokens):
            score += 0.5
        if any(tok in triple['tail'].lower() for tok in q_tokens):
            score += 0.5
        item = dict(triple)
        item['score'] = round(float(score), 4)
        scored.append(item)

    scored.sort(key=lambda x: x['score'], reverse=True)
    selected = scored[: max(1, top_k)]
    selected_nodes = {x['head'] for x in selected} | {x['tail'] for x in selected}

    neighbors = []
    seen = {(x['head'], x['relation'], x['tail']) for x in selected}
    for triple in triples:
        key = (triple['head'], triple['relation'], triple['tail'])
        if key in seen:
            continue
        if triple['head'] in selected_nodes or triple['tail'] in selected_nodes:
            neighbors.append(dict(triple))
        if len(neighbors) >= max(2, top_k):
            break

    yaml_text = memory_subgraph_to_yaml(selected, neighbors)
    return {
        'selected': selected,
        'neighbors': neighbors,
        'yaml': yaml_text,
    }


def memory_subgraph_to_yaml(selected: list[dict], neighbors: list[dict]) -> str:
    """把选中的记忆子图组织成 YAML 风格文本。"""
    if not selected and not neighbors:
        return ''

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in selected:
        grouped['selected'].append(item)
    for item in neighbors:
        grouped['neighbors'].append(item)

    lines = ['memory_graph:']
    for group_name in ('selected', 'neighbors'):
        lines.append(f'  {group_name}:')
        if not grouped[group_name]:
            lines.append('    []')
            continue
        for item in grouped[group_name]:
            lines.append('    - head: ' + json.dumps(item.get('head', ''), ensure_ascii=False))
            lines.append('      relation: ' + json.dumps(item.get('relation', ''), ensure_ascii=False))
            lines.append('      tail: ' + json.dumps(item.get('tail', ''), ensure_ascii=False))
            if item.get('score') is not None:
                lines.append(f"      score: {item.get('score')}")
            if item.get('chunk_id') is not None:
                lines.append(f"      chunk_id: {item.get('chunk_id')}")
    return '\n'.join(lines)


def build_memory_context_chunk(question: str, algorithm_id: int | None = None, top_k: int = 6) -> dict | None:
    """把记忆子图包装成一个可直接参与推理的伪 chunk。"""
    subgraph = select_memory_subgraph(question, algorithm_id=algorithm_id, top_k=top_k)
    if not subgraph.get('yaml'):
        return None
    return {
        'id': -3,
        'algorithm_id': algorithm_id or 0,
        'chunk_type': 'memory_graph',
        'title': '结构化记忆子图',
        'content': subgraph['yaml'],
        'source': 'memory_graph',
        'score': 1.0,
        'selected_memory': subgraph.get('selected', []),
    }
