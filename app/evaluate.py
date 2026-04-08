"""自动评测模块。

负责构造评测流程，并比较不同问答方式的效果。"""


from typing import Any

from .db_source import fetch_eval_samples
from .deepseek import deepseek_chat
from .metrics import mean_scores, score_answer
from .rag import retrieve
from .tools import build_tool_chunks


def _answer_from_refs(question: str, refs: list[dict], model_mode: str) -> str:
    """根据检索片段生成回答；可选择直接抽取式输出或交给大模型整理。"""
    if model_mode == 'extractive':
        lines = []
        for r in refs[:4]:
            lines.append(r.get('content') or '')
        return '\n'.join(lines)[:1800]

    context = []
    for r in refs:
        context.append(
            f"[chunk_id={r.get('id')};algorithm_id={r.get('algorithm_id')};type={r.get('chunk_type')};source={r.get('source')}]\n{r.get('content')}"
        )

    return deepseek_chat(
        [
            {
                'role': 'system',
                'content': '你是算法助教。仅基于给定片段回答，尽量简洁准确。',
            },
            {'role': 'user', 'content': f"问题：{question}\n\n片段：\n\n" + '\n\n'.join(context)},
        ],
        temperature=0.2,
        max_tokens=1200,
    )


def evaluate_rag(
    limit: int = 12,
    top_k: int = 5,
    algorithm_id: int | None = None,
    retrieval_mode: str = 'hybrid',
) -> dict[str, Any]:
    """运行评测主流程，对比 baseline_extractive、rag_basic 和 rag_plus。"""
    samples = fetch_eval_samples(limit=limit, algorithm_id=algorithm_id)
    if not samples:
        return {
            'summary': {'total': 0, 'done': 0, 'failed': 0},
            'methods': {},
            'cases': [],
            'message': '没有可评测样本',
        }

    method_scores: dict[str, list[dict]] = {
        'baseline_extractive': [],
        'rag_basic': [],
        'rag_plus': [],
    }
    cases = []
    failed = 0

    for s in samples:
        q = s['question']
        ref = s['reference_answer']
        aid = s['algorithm_id']
        case_item: dict[str, Any] = {
            'sample_id': s['sample_id'],
            'algorithm_id': aid,
            'algorithm_name': s['algorithm_name'],
            'question': q,
            'source_title': s['source_title'],
            'scores': {},
        }

        try:
            # 1) Baseline: no expansion/tools/memory + extractive output
            refs_base = retrieve(
                q,
                top_k=top_k,
                algorithm_id=aid,
                mode=retrieval_mode,
                use_expansion=False,
                include_memory=False,
            )
            pred_base = _answer_from_refs(q, refs_base, model_mode='extractive')
            m_base = score_answer(pred_base, ref)
            method_scores['baseline_extractive'].append(m_base)
            case_item['scores']['baseline_extractive'] = m_base

            # 2) Basic RAG: no expansion/tools/memory + LLM generation
            pred_basic = _answer_from_refs(q, refs_base, model_mode='llm')
            m_basic = score_answer(pred_basic, ref)
            method_scores['rag_basic'].append(m_basic)
            case_item['scores']['rag_basic'] = m_basic

            # 3) Enhanced RAG: expansion + tools (memory off for fairness)
            refs_plus = retrieve(
                q,
                top_k=top_k,
                algorithm_id=aid,
                mode=retrieval_mode,
                use_expansion=True,
                include_memory=False,
            )
            refs_plus_all = refs_plus + build_tool_chunks(q, refs_plus, algorithm_id=aid)
            pred_plus = _answer_from_refs(q, refs_plus_all, model_mode='llm')
            m_plus = score_answer(pred_plus, ref)
            method_scores['rag_plus'].append(m_plus)
            case_item['scores']['rag_plus'] = m_plus
        except Exception as e:
            failed += 1
            case_item['error'] = str(e)

        cases.append(case_item)

    methods = {name: mean_scores(vals) for name, vals in method_scores.items()}
    done = len(samples) - failed
    return {
        'summary': {'total': len(samples), 'done': done, 'failed': failed, 'retrieval_mode': retrieval_mode},
        'methods': methods,
        'cases': cases,
    }
