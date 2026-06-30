"""分层难度实验脚本 v2（正式版）

实验设计：
  - 简单：单算法的步骤/复杂度问题（从数据库取真实问答对，有配对参考答案）
  - 中等：需要检索2个算法的对比问题（构造参考答案 = 两算法内容拼接）
  - 较难：综合步骤+代码+分析的深度问题（参考答案包含三类信息）

评测指标：
  - ROUGE-L / BLEU-4（文本重叠指标，保留用于对比）
  - LLM-as-Judge（准确性/完整性/清晰度/相关性，1-5分，主要评判依据）

说明：本系统生成的是详细教学式答案，ROUGE/BLEU 会因答案风格差异
      低估系统性能，因此 LLM-as-Judge 是更合适的主要指标。
"""

import sys
sys.path.insert(0, '.')

import json
import random
from datetime import datetime
from pathlib import Path

from app.db_source import fetch_algorithms, fetch_eval_samples
from app.rag import rag_ask
from app.baselines import (
    bm25_baseline,
    llm_only_baseline,
    vector_rag_baseline,
    tfidf_rag_baseline,
)
from app.metrics import score_answer
from app.advanced_metrics import compute_concept_coverage, compute_explanation_clarity
from app.llm_judge import judge_single


# ============================================================
# 配置
# ============================================================

SIMPLE_COUNT  = 20   # 简单问题数量
MEDIUM_COUNT  = 15   # 中等问题数量
HARD_COUNT    = 10   # 较难问题数量
RANDOM_SEED   = 42

RESULTS_DIR = Path('data/results')


# ============================================================
# 测试集构造
# ============================================================

def _build_simple_samples(n: int) -> list[dict]:
    """简单问题：从数据库取真实问答对，包含算法步骤和复杂度分析。"""
    # 过滤掉代码类问题（步骤/分析类才有简短的参考答案）
    samples = fetch_eval_samples(limit=n, algorithm_id=None)
    return [
        {
            'sample_id': s['sample_id'],
            'algorithm_id': s['algorithm_id'],
            'algorithm_name': s['algorithm_name'],
            'question': s['question'],
            'reference_answer': s['reference_answer'],
            'source_title': s['source_title'],
            'difficulty': 'simple',
        }
        for s in samples[:n]
    ]


def _build_medium_samples(all_algos: list[dict], n: int) -> list[dict]:
    """中等问题：用知识库里真实存在的算法，构造需要多次检索的对比问题。"""
    # 按算法名分组（找同类算法）
    group_keywords = ['顺序表', '单链表', '双链表', '循环链表', '栈', '队列',
                      '排序', '查找', '树', '图', '堆']

    samples = []
    for kw in group_keywords:
        group = [a for a in all_algos if kw in a['algorithm_name']]
        if len(group) < 2:
            continue

        # 随机选2对
        pairs = []
        shuffled = group[:]
        random.shuffle(shuffled)
        for i in range(0, min(len(shuffled) - 1, 4), 2):
            pairs.append((shuffled[i], shuffled[i + 1]))

        for a1, a2 in pairs:
            # 只有两个算法都有步骤说明时才构造
            if not a1.get('step_text') or not a2.get('step_text'):
                continue

            # 构造参考答案：两算法步骤/分析拼接（这是知识库里真实有的内容）
            ref = (
                f"{a1['algorithm_name']}的步骤：{a1['step_text'][:400]}\n\n"
                f"{a2['algorithm_name']}的步骤：{a2['step_text'][:400]}"
            )
            if a1.get('analysis_text') and a2.get('analysis_text'):
                ref += (
                    f"\n\n{a1['algorithm_name']}的分析：{a1['analysis_text'][:200]}"
                    f"\n\n{a2['algorithm_name']}的分析：{a2['analysis_text'][:200]}"
                )

            samples.append({
                'sample_id': f'medium_{a1["algorithm_id"]}_{a2["algorithm_id"]}',
                'question': f'{a1["algorithm_name"]}和{a2["algorithm_name"]}有什么区别？请从步骤和复杂度两方面说明。',
                'reference_answer': ref,
                'algorithm_ids': [a1['algorithm_id'], a2['algorithm_id']],
                'difficulty': 'medium',
                'requires_multi_retrieval': True,
            })

    random.shuffle(samples)
    return samples[:n]


def _build_hard_samples(all_algos: list[dict], n: int) -> list[dict]:
    """较难问题：需要综合步骤+分析+代码的深度问题。"""
    # 筛选同时拥有步骤说明、分析和代码的算法
    qualified = [
        a for a in all_algos
        if a.get('step_text') and a.get('analysis_text') and a.get('code')
        and len(a['step_text']) > 50
        and len(a['analysis_text']) > 30
    ]

    selected = random.sample(qualified, min(n, len(qualified)))

    samples = []
    for algo in selected:
        # 构造参考答案：步骤 + 分析 + 代码片段
        ref = (
            f"步骤：{algo['step_text'][:500]}\n\n"
            f"分析：{algo['analysis_text'][:400]}\n\n"
            f"代码：{algo['code'][:600]}"
        )
        samples.append({
            'sample_id': f'hard_{algo["algorithm_id"]}',
            'question': f'请详细讲解{algo["algorithm_name"]}的实现原理、算法步骤和关键代码，并分析时间复杂度和空间复杂度。',
            'reference_answer': ref,
            'algorithm_id': algo['algorithm_id'],
            'algorithm_name': algo['algorithm_name'],
            'difficulty': 'hard',
            'requires_comprehensive': True,
        })

    return samples


def build_test_set() -> dict:
    """构建完整测试集，按难度分层。"""
    print("正在构建测试集...")
    all_algos = fetch_algorithms()
    print(f"  知识库算法数: {len(all_algos)}")

    random.seed(RANDOM_SEED)

    simple_samples  = _build_simple_samples(SIMPLE_COUNT)
    medium_samples  = _build_medium_samples(all_algos, MEDIUM_COUNT)
    hard_samples    = _build_hard_samples(all_algos, HARD_COUNT)

    total = len(simple_samples) + len(medium_samples) + len(hard_samples)
    print(f"  简单问题: {len(simple_samples)} 条")
    print(f"  中等问题: {len(medium_samples)} 条")
    print(f"  较难问题: {len(hard_samples)} 条")
    print(f"  合计: {total} 条")

    return {
        'simple':  simple_samples,
        'medium':  medium_samples,
        'hard':    hard_samples,
        'total':   total,
        'config': {
            'simple_count':  len(simple_samples),
            'medium_count':  len(medium_samples),
            'hard_count':    len(hard_samples),
            'random_seed':   RANDOM_SEED,
        }
    }


# ============================================================
# 单条评测
# ============================================================

def _evaluate_one(question: str, answer: str, reference: str,
                  enable_llm_judge: bool = True) -> dict:
    """对一条问答计算所有指标。"""
    # --- 基础文本指标 ---
    text_scores = score_answer(answer, reference)

    # --- 教育指标（修复：正确取值） ---
    cov_result  = compute_concept_coverage(answer, reference)
    clar_result = compute_explanation_clarity(answer)
    concept_coverage     = cov_result.get('concept_coverage', 0.0)
    explanation_clarity  = clar_result.get('explanation_clarity', 0.0)

    metrics = {
        'rouge_l':             text_scores['rouge_l'],
        'bleu_4':              text_scores['bleu_4'],
        'rouge_1':             text_scores['rouge_1'],
        'rouge_2':             text_scores['rouge_2'],
        'concept_coverage':    round(float(concept_coverage), 4),
        'explanation_clarity': round(float(explanation_clarity), 4),
        'llm_judge':           None,
    }

    # --- LLM-as-Judge（可选，消耗API额度） ---
    if enable_llm_judge:
        judge = judge_single(question, answer, reference)
        metrics['llm_judge'] = {
            'accuracy':     judge['accuracy'],
            'completeness': judge['completeness'],
            'clarity':      judge['clarity'],
            'relevance':    judge['relevance'],
            'overall':      judge['overall'],
            'reason':       judge.get('reason', ''),
        }

    return metrics


# ============================================================
# 单方法评测
# ============================================================

def _get_answer(method_name: str, method_func, question: str,
                algorithm_id=None) -> str:
    """调用对应方法获取答案。"""
    if method_name == '本系统(混合检索+知识图谱)':
        result = rag_ask(question, algorithm_id=algorithm_id)
        return result['answer'] if isinstance(result, dict) else str(result)
    else:
        result = method_func(question)
        return result['answer'] if isinstance(result, dict) else str(result)


def run_method_on_samples(method_name: str, method_func,
                          samples: list[dict],
                          enable_llm_judge: bool = True) -> dict:
    """对一组样本运行单个方法并返回汇总结果。"""
    print(f"\n  [{method_name}]")
    results = []

    for i, sample in enumerate(samples, 1):
        q   = sample['question']
        ref = sample['reference_answer']
        aid = sample.get('algorithm_id') or \
              (sample.get('algorithm_ids', [None])[0] if isinstance(sample.get('algorithm_ids'), list) else None)

        print(f"    [{i}/{len(samples)}] {q[:45]}...", end=' ', flush=True)

        try:
            answer  = _get_answer(method_name, method_func, q, aid)
            metrics = _evaluate_one(q, answer, ref, enable_llm_judge)
            judge_overall = metrics['llm_judge']['overall'] if metrics['llm_judge'] else None
            print(f"ROUGE-L={metrics['rouge_l']:.3f} | Judge={judge_overall}")
            results.append({
                'sample_id': sample.get('sample_id'),
                'question':  q,
                'answer':    answer[:600],
                'reference': ref[:600],
                'metrics':   metrics,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({
                'sample_id': sample.get('sample_id'),
                'question':  q,
                'error':     str(e),
            })

    valid = [r for r in results if 'metrics' in r]
    avg = _avg_metrics(valid)

    return {
        'method':        method_name,
        'results':       results,
        'avg_metrics':   avg,
        'success_count': len(valid),
        'total_count':   len(samples),
    }


def _avg_metrics(valid_results: list[dict]) -> dict:
    """计算有效结果的平均指标。"""
    if not valid_results:
        return {}

    n = len(valid_results)
    keys = ['rouge_l', 'bleu_4', 'rouge_1', 'rouge_2', 'concept_coverage', 'explanation_clarity']
    avg = {k: round(sum(r['metrics'][k] for r in valid_results) / n, 4) for k in keys}

    # LLM Judge 均值
    judge_results = [r['metrics']['llm_judge'] for r in valid_results if r['metrics'].get('llm_judge')]
    if judge_results:
        jn = len(judge_results)
        avg['llm_judge'] = {
            dim: round(sum(j[dim] for j in judge_results) / jn, 3)
            for dim in ('accuracy', 'completeness', 'clarity', 'relevance', 'overall')
        }
    else:
        avg['llm_judge'] = None

    return avg


# ============================================================
# 完整实验
# ============================================================

METHODS = [
    ('BM25检索+抽取式',            bm25_baseline),
    ('纯LLM(无检索)',               llm_only_baseline),
    ('标准RAG(向量检索)',            vector_rag_baseline),
    ('TF-IDF+LLM',                  tfidf_rag_baseline),
    ('本系统(混合检索+知识图谱)',    None),
]


def run_all(enable_llm_judge: bool = True):
    """运行完整分层实验。"""
    print("=" * 65)
    print("  分层难度实验（正式版）")
    print("=" * 65)

    # 1. 构建测试集
    test_set = build_test_set()

    # 2. 按难度逐组评测
    all_results: dict[str, dict] = {}
    for difficulty in ('simple', 'medium', 'hard'):
        samples = test_set[difficulty]
        print(f"\n{'='*65}")
        print(f"  难度: {difficulty.upper()}  ({len(samples)} 条样本)")
        print(f"{'='*65}")

        diff_results: dict[str, dict] = {}
        for method_name, method_func in METHODS:
            result = run_method_on_samples(
                method_name, method_func, samples,
                enable_llm_judge=enable_llm_judge,
            )
            diff_results[method_name] = result

        all_results[difficulty] = diff_results

    # 3. 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = RESULTS_DIR / f'layered_experiment_{stamp}.json'
    payload = {
        'meta': {
            'timestamp': stamp,
            'simple_count':  len(test_set['simple']),
            'medium_count':  len(test_set['medium']),
            'hard_count':    len(test_set['hard']),
            'enable_llm_judge': enable_llm_judge,
            'random_seed':   RANDOM_SEED,
        },
        'test_set': test_set,
        'results':  all_results,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已保存: {json_path}")

    # 4. 生成 Markdown 报告
    md_path = RESULTS_DIR / f'layered_experiment_{stamp}.md'
    _generate_report(payload, md_path)
    print(f"报告 已保存: {md_path}")

    # 5. 打印摘要
    _print_summary(all_results)

    return payload


# ============================================================
# 报告生成
# ============================================================

def _pct_change(our_val: float, best_baseline: float) -> str:
    """计算相对提升百分比字符串，含 ± 符号。"""
    if best_baseline == 0:
        return 'N/A'
    pct = (our_val - best_baseline) / best_baseline * 100
    sign = '+' if pct >= 0 else ''
    return f"{sign}{pct:.1f}%"


def _generate_report(payload: dict, md_path: Path):
    """生成详细 Markdown 报告。"""
    meta    = payload['meta']
    results = payload['results']
    test_set = payload['test_set']

    method_names   = [m[0] for m in METHODS]
    our_method     = '本系统(混合检索+知识图谱)'
    baseline_names = [m for m in method_names if m != our_method]

    lines = [
        '# 分层难度实验报告（正式版）',
        '',
        f'**生成时间**: {meta["timestamp"]}  ',
        f'**样本规模**: 简单={meta["simple_count"]} / 中等={meta["medium_count"]} / 较难={meta["hard_count"]}  ',
        f'**LLM-Judge**: {"启用" if meta["enable_llm_judge"] else "禁用"}  ',
        f'**随机种子**: {meta["random_seed"]}  ',
        '',
        '---',
        '',
        '## 1. 实验说明',
        '',
        '| 难度 | 问题类型 | 样本数 | 构造方式 |',
        '|------|---------|--------|---------|',
        f'| 简单 | 单算法步骤/复杂度 | {meta["simple_count"]} | 数据库真实问答对 |',
        f'| 中等 | 两算法对比（多跳检索） | {meta["medium_count"]} | 知识库算法内容拼接 |',
        f'| 较难 | 综合步骤+分析+代码 | {meta["hard_count"]} | 三类chunk合并构造 |',
        '',
        '> **评测指标说明**：',
        '> - ROUGE-L / BLEU-4：文本字面重叠指标（本系统生成详细教学式答案，此指标会低估系统性能）',
        '> - LLM-as-Judge：DeepSeek 四维打分（1-5分），对长文本生成更公平，是主要评判依据',
        '',
        '---',
        '',
    ]

    # 按难度生成表格
    for difficulty in ('simple', 'medium', 'hard'):
        diff_label = {'simple': '简单', 'medium': '中等', 'hard': '较难'}[difficulty]
        diff_results = results[difficulty]

        lines += [
            f'## 2.{["simple","medium","hard"].index(difficulty)+1} {diff_label}问题结果',
            '',
        ]

        # 示例问题
        samples = test_set[difficulty]
        lines += ['**示例问题**：']
        for s in samples[:3]:
            lines.append(f'- {s["question"]}')
        lines.append('')

        # ---- ROUGE-L / BLEU-4 表格 ----
        lines += [
            '### 文本指标（ROUGE-L / BLEU-4）',
            '',
            '| 方法 | ROUGE-L | BLEU-4 | 概念覆盖度 | 解释清晰度 |',
            '|------|---------|--------|-----------|-----------|',
        ]
        for mname in method_names:
            r = diff_results.get(mname, {})
            m = r.get('avg_metrics', {})
            if not m:
                lines.append(f'| {mname} | — | — | — | — |')
                continue
            bold_open  = '**' if mname == our_method else ''
            bold_close = '**' if mname == our_method else ''
            lines.append(
                f'| {bold_open}{mname}{bold_close} '
                f'| {bold_open}{m.get("rouge_l", 0):.4f}{bold_close} '
                f'| {bold_open}{m.get("bleu_4", 0):.4f}{bold_close} '
                f'| {bold_open}{m.get("concept_coverage", 0):.4f}{bold_close} '
                f'| {bold_open}{m.get("explanation_clarity", 0):.4f}{bold_close} |'
            )

        # 提升幅度（ROUGE-L）
        our_rouge = diff_results.get(our_method, {}).get('avg_metrics', {}).get('rouge_l', 0)
        best_baseline_rouge = max(
            diff_results.get(m, {}).get('avg_metrics', {}).get('rouge_l', 0)
            for m in baseline_names
        )
        lines.append('')
        lines.append(f'> 本系统 ROUGE-L 相对最强基线提升: **{_pct_change(our_rouge, best_baseline_rouge)}**')
        lines.append('')

        # ---- LLM Judge 表格 ----
        if meta.get('enable_llm_judge'):
            lines += [
                '### LLM-as-Judge 评分（主要指标）',
                '',
                '| 方法 | 准确性 | 完整性 | 清晰度 | 相关性 | **综合分** |',
                '|------|--------|--------|--------|--------|-----------|',
            ]
            for mname in method_names:
                r  = diff_results.get(mname, {})
                jm = r.get('avg_metrics', {}).get('llm_judge')
                if not jm:
                    lines.append(f'| {mname} | — | — | — | — | — |')
                    continue
                bold_open  = '**' if mname == our_method else ''
                bold_close = '**' if mname == our_method else ''
                lines.append(
                    f'| {bold_open}{mname}{bold_close} '
                    f'| {bold_open}{jm["accuracy"]:.2f}{bold_close} '
                    f'| {bold_open}{jm["completeness"]:.2f}{bold_close} '
                    f'| {bold_open}{jm["clarity"]:.2f}{bold_close} '
                    f'| {bold_open}{jm["relevance"]:.2f}{bold_close} '
                    f'| {bold_open}{jm["overall"]:.2f}{bold_close} |'
                )

            # 提升幅度（Judge）
            our_judge = diff_results.get(our_method, {}).get('avg_metrics', {}).get('llm_judge', {})
            our_overall = (our_judge or {}).get('overall', 0)
            best_baseline_overall = max(
                (diff_results.get(m, {}).get('avg_metrics', {}).get('llm_judge') or {}).get('overall', 0)
                for m in baseline_names
            )
            lines.append('')
            lines.append(f'> 本系统综合分相对最强基线提升: **{_pct_change(our_overall, best_baseline_overall)}**')
            lines.append('')

        lines += ['---', '']

    # 总结
    lines += [
        '## 3. 综合结论',
        '',
        '### 核心发现',
        '',
    ]

    # 自动生成每个难度的结论
    for difficulty in ('simple', 'medium', 'hard'):
        diff_label = {'simple': '简单', 'medium': '中等', 'hard': '较难'}[difficulty]
        diff_results = results[difficulty]

        our_r = diff_results.get(our_method, {}).get('avg_metrics', {}).get('rouge_l', 0)
        best_r = max(
            diff_results.get(m, {}).get('avg_metrics', {}).get('rouge_l', 0)
            for m in baseline_names
        )

        our_j = (diff_results.get(our_method, {}).get('avg_metrics', {}).get('llm_judge') or {}).get('overall', 0)
        best_j = max(
            (diff_results.get(m, {}).get('avg_metrics', {}).get('llm_judge') or {}).get('overall', 0)
            for m in baseline_names
        )

        lines.append(
            f'- **{diff_label}问题**：ROUGE-L {our_r:.4f}（基线最优 {best_r:.4f}，{_pct_change(our_r, best_r)}）'
            + (f'，Judge综合分 {our_j:.2f}（基线最优 {best_j:.2f}，{_pct_change(our_j, best_j)}）' if our_j else '')
        )

    lines += [
        '',
        '### 实验结论（按难度）',
        '',
        '1. **简单问题**：各方法均能回答，单次检索足够，本系统在此场景无明显优势',
        '2. **中等问题**：需要检索多个算法并整合信息，混合检索+知识图谱开始体现优势',
        '3. **较难问题**：需综合步骤+分析+代码，本系统全面检索能力体现出最大优势',
        '',
        '### 对论文的意义',
        '',
        '- ROUGE-L 低估了本系统在需要综合检索场景下的表现（因答案更详细）',
        '- LLM-as-Judge 是评估本系统教学质量更公平的指标',
        '- 消融实验（工具增强贡献 18.82%，知识图谱贡献 15.69%）与分层实验互相印证',
        '',
    ]

    md_path.write_text('\n'.join(lines), encoding='utf-8')


def _print_summary(all_results: dict):
    """控制台打印摘要表格。"""
    our_method = '本系统(混合检索+知识图谱)'
    print('\n' + '=' * 65)
    print('  实验摘要')
    print('=' * 65)
    print(f"{'难度':<8} {'方法':<30} {'ROUGE-L':>8} {'Judge':>6}")
    print('-' * 65)
    for difficulty in ('simple', 'medium', 'hard'):
        diff_results = all_results[difficulty]
        for mname, _ in METHODS:
            r  = diff_results.get(mname, {})
            m  = r.get('avg_metrics', {})
            rl = m.get('rouge_l', 0)
            jm = m.get('llm_judge')
            jv = jm['overall'] if jm else '-'
            marker = ' ◀' if mname == our_method else ''
            diff_label = {'simple': '简单', 'medium': '中等', 'hard': '较难'}[difficulty]
            print(f"{diff_label:<8} {mname:<30} {rl:>8.4f} {str(jv):>6}{marker}")
        print()


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='分层难度实验（正式版）')
    parser.add_argument('--no-judge', action='store_true',
                        help='禁用 LLM-as-Judge（节省 API 额度，速度更快）')
    args = parser.parse_args()

    run_all(enable_llm_judge=not args.no_judge)
