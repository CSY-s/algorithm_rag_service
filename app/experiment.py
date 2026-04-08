"""实验与结果导出模块。

负责统计系统状态、运行批量实验，并把结果写入报告文件。"""


import json
import os
from datetime import datetime
from typing import Any

from .db_source import fetch_eval_samples
from .evaluate import evaluate_rag
from .hybrid_retrieval import RETRIEVAL_MODES
from .rag import retrieve
from .store import get_chunks


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')


def _ensure_results_dir() -> str:
    """确保实验结果目录存在。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def get_system_stats() -> dict[str, Any]:
    """统计当前知识库的 chunk 数量、来源分布等信息。"""
    rows = get_chunks()
    chunk_types: dict[str, int] = {}
    sources: dict[str, int] = {}
    algorithm_ids = set()

    for row in rows:
        ctype = str(row.get('chunk_type') or 'unknown')
        source = str(row.get('source') or 'unknown')
        chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        if row.get('algorithm_id') is not None:
            algorithm_ids.add(int(row.get('algorithm_id') or 0))

    return {
        'total_chunks': len(rows),
        'algorithm_count': len([x for x in algorithm_ids if x > 0]),
        'chunk_types': chunk_types,
        'sources': sources,
        'retrieval_modes': sorted(RETRIEVAL_MODES),
    }


def search_only(
    question: str,
    top_k: int = 5,
    algorithm_id: int | None = None,
    retrieval_mode: str = 'hybrid',
    use_expansion: bool = True,
    include_memory: bool = True,
) -> dict[str, Any]:
    """只执行检索逻辑，用于单独观察召回结果。"""
    refs = retrieve(
        question,
        top_k=top_k,
        algorithm_id=algorithm_id,
        mode=retrieval_mode,
        use_expansion=use_expansion,
        include_memory=include_memory,
    )
    return {
        'question': question,
        'count': len(refs),
        'retrieval_mode': retrieval_mode,
        'results': refs,
    }


def _summarize_methods(methods: dict[str, dict[str, float]]) -> dict[str, float]:
    """把多种方法的平均分再汇总成一份总分概览。"""
    if not methods:
        return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0, 'bleu_4': 0.0}
    keys = ['rouge_1', 'rouge_2', 'rouge_l', 'bleu_4']
    out = {}
    for key in keys:
        values = [float(item.get(key, 0.0)) for item in methods.values()]
        out[key] = round(sum(values) / max(1, len(values)), 6)
    return out


def _make_report(data: dict[str, Any]) -> str:
    """把实验结果整理成 Markdown 报告。"""
    lines = [
        '# RAG Experiment Report',
        '',
        f"- time: {data['meta']['timestamp']}",
        f"- samples: {data['meta']['limit']}",
        f"- algorithm_id: {data['meta']['algorithm_id']}",
        '',
        '## Config Comparison',
        '',
        '| retrieval_mode | top_k | rouge_1 | rouge_2 | rouge_l | bleu_4 | done | failed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]

    for item in data['experiments']:
        avg = item['avg_methods']
        summary = item['summary']
        lines.append(
            f"| {item['retrieval_mode']} | {item['top_k']} | {avg['rouge_1']:.4f} | {avg['rouge_2']:.4f} | "
            f"{avg['rouge_l']:.4f} | {avg['bleu_4']:.4f} | {summary['done']} | {summary['failed']} |"
        )

    lines.extend(
        [
            '',
            '## Notes',
            '',
            '- `avg_methods` is the mean of `baseline_extractive`, `rag_basic`, and `rag_plus`.',
            '- Use the full JSON output for per-method and per-case details.',
        ]
    )
    return '\n'.join(lines)


def run_experiment_suite(
    limit: int = 8,
    algorithm_id: int | None = None,
    top_k_values: list[int] | None = None,
    retrieval_modes: list[str] | None = None,
) -> dict[str, Any]:
    """批量遍历不同参数组合，并输出实验结果与最佳配置。"""
    top_k_values = top_k_values or [3, 5, 8]
    retrieval_modes = retrieval_modes or ['tfidf', 'keyword', 'vector', 'hybrid']
    retrieval_modes = [m for m in retrieval_modes if m in RETRIEVAL_MODES]
    if not retrieval_modes:
        retrieval_modes = ['hybrid']

    samples = fetch_eval_samples(limit=limit, algorithm_id=algorithm_id)
    if not samples:
        return {
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'limit': limit,
                'algorithm_id': algorithm_id,
            },
            'experiments': [],
            'message': '没有可实验样本',
        }

    experiments = []
    for retrieval_mode in retrieval_modes:
        for top_k in top_k_values:
            result = evaluate_rag(
                limit=limit,
                top_k=top_k,
                algorithm_id=algorithm_id,
                retrieval_mode=retrieval_mode,
            )
            experiments.append(
                {
                    'retrieval_mode': retrieval_mode,
                    'top_k': top_k,
                    'summary': result.get('summary', {}),
                    'methods': result.get('methods', {}),
                    'avg_methods': _summarize_methods(result.get('methods', {})),
                }
            )

    best = max(
        experiments,
        key=lambda item: (
            item['methods'].get('rag_plus', {}).get('rouge_l', 0.0),
            item['methods'].get('rag_plus', {}).get('bleu_4', 0.0),
        ),
    )

    payload = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'limit': limit,
            'algorithm_id': algorithm_id,
            'sample_count': len(samples),
        },
        'experiments': experiments,
        'best': best,
    }

    _ensure_results_dir()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(RESULTS_DIR, f'experiment_suite_{stamp}.json')
    md_path = os.path.join(RESULTS_DIR, f'experiment_suite_{stamp}.md')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(_make_report(payload))

    payload['artifacts'] = {'json': json_path, 'markdown': md_path}
    return payload
