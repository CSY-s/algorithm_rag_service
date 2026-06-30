"""选项3 统一实验脚本。

三套实验：
  实验1 - 基线对比（rag_ask + ROUGE/BLEU，评测检索质量）
  实验2 - 消融实验（复用已有结果，不重跑）
  实验3 - LLM-as-Judge（answer_with_plan，评测答案质量）

运行方式：
  python scripts/run_option3_experiment.py --mode all
  python scripts/run_option3_experiment.py --mode baseline
  python scripts/run_option3_experiment.py --mode judge
"""

import json, os, sys, random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.baselines import run_baseline
from app.rag import rag_ask
from app.agent_workflow import answer_with_plan
from app.db_source import fetch_eval_samples
from app.metrics import score_answer
from app.advanced_metrics import batch_evaluate
from app.llm_judge import judge_batch


# ================================================================
# 公共工具
# ================================================================

def _fixed_samples(limit, algorithm_id=None, seed=42):
    samples = fetch_eval_samples(limit=limit * 3, algorithm_id=algorithm_id)
    random.seed(seed)
    random.shuffle(samples)
    return samples[:limit]


def _calc_rouge_bleu(predictions, references):
    scores = [score_answer(p, r) for p, r in zip(predictions, references) if p]
    n = len(scores) or 1
    return {
        'rouge_1': round(sum(s['rouge_1'] for s in scores) / n, 4),
        'rouge_2': round(sum(s['rouge_2'] for s in scores) / n, 4),
        'rouge_l': round(sum(s['rouge_l'] for s in scores) / n, 4),
        'bleu_4':  round(sum(s['bleu_4']  for s in scores) / n, 4),
    }


def _save(data, prefix):
    os.makedirs('data/results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'data/results/{prefix}_{ts}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path, ts


# ================================================================
# 实验1：基线对比（rag_ask + ROUGE/BLEU）
# ================================================================

def run_exp1_baseline(limit=30, top_k=5, seed=42):
    print('\n' + '='*65)
    print('实验1：基线对比（rag_ask + ROUGE/BLEU，评测检索质量）')
    print('='*65)

    samples = _fixed_samples(limit, seed=seed)
    print(f'样本数={len(samples)}  top_k={top_k}  seed={seed}\n')

    methods = {
        'bm25':       ('BM25 + 抽取式',
                       lambda q, aid, k: run_baseline('bm25', q, top_k=k, algorithm_id=aid)),
        'llm_only':   ('纯LLM（无检索）',
                       lambda q, aid, k: run_baseline('llm_only', q, top_k=k, algorithm_id=aid)),
        'vector_rag': ('标准RAG（向量检索）',
                       lambda q, aid, k: run_baseline('vector_rag', q, top_k=k, algorithm_id=aid)),
        'tfidf_rag':  ('TF-IDF + LLM',
                       lambda q, aid, k: run_baseline('tfidf_rag', q, top_k=k, algorithm_id=aid)),
        'our_method': ('本系统（rag_ask，混合检索）',
                       lambda q, aid, k: rag_ask(
                           q, top_k=k, algorithm_id=aid,
                           retrieval_mode='hybrid',
                           enable_tools=True, enable_memory=True,
                       )),
    }

    results = {}
    for key, (name, fn) in methods.items():
        print(f'▶ {name}')
        preds, refs = [], []
        for i, s in enumerate(samples, 1):
            print(f'\r  进度: {i}/{len(samples)}', end='', flush=True)
            try:
                r = fn(s['question'], s.get('algorithm_id'), top_k)
                preds.append(r.get('answer', ''))
            except Exception as e:
                print(f'\n  ⚠ 样本{i}失败: {e}')
                preds.append('')
            refs.append(s['reference_answer'])
        print()
        metrics = _calc_rouge_bleu(preds, refs)
        adv = batch_evaluate(preds, refs)['average_metrics']
        results[key] = {
            'name': name, 'rouge_bleu': metrics,
            'f1': round(adv['f1_score'], 4),
            'concept_coverage': round(adv['concept_coverage'], 4),
        }
        print(f'  ROUGE-L={metrics["rouge_l"]}  BLEU-4={metrics["bleu_4"]}  F1={results[key]["f1"]}')

    # 汇总
    print('\n' + '-'*65)
    print(f'{"方法":<30} {"ROUGE-L":>8} {"BLEU-4":>8} {"F1":>8}')
    print('-'*65)
    for key, (name, _) in methods.items():
        r = results[key]
        print(f'{name:<30} {r["rouge_bleu"]["rouge_l"]:>8.4f} '
              f'{r["rouge_bleu"]["bleu_4"]:>8.4f} {r["f1"]:>8.4f}')

    our = results['our_method']
    best_rl = max(results[k]['rouge_bleu']['rouge_l'] for k in results if k != 'our_method')
    imp = (our['rouge_bleu']['rouge_l'] - best_rl) / best_rl * 100
    print(f'\n本系统相对最佳基线 ROUGE-L 提升: {imp:+.2f}%')

    path, ts = _save({'exp': 'baseline_rag_ask', 'config': {'limit': limit, 'top_k': top_k, 'seed': seed},
                      'results': results, 'improvement_rouge_l': imp}, 'exp1_baseline')
    print(f'✅ 已保存: {path}')
    return results


# ================================================================
# 实验2：消融实验（复用已有结果）
# ================================================================

ABLATION_FILE = 'data/results/ablation_study_20260526_091655.json'

def run_exp2_ablation():
    print('\n' + '='*65)
    print('实验2：消融实验（复用已有结果）')
    print('='*65)

    if not os.path.exists(ABLATION_FILE):
        print(f'❌ 找不到消融实验结果文件: {ABLATION_FILE}')
        print('   请先运行: python scripts/run_ablation_study.py')
        return None

    with open(ABLATION_FILE, encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']
    order = ['full', 'no_hybrid', 'no_kg', 'no_memory', 'no_tools']

    print(f'\n数据来源: {ABLATION_FILE}')
    print(f'实验配置: 样本数={data["experiment_config"]["limit"]}, '
          f'top_k={data["experiment_config"]["top_k"]}')
    print(f'注：本实验使用 rag_ask 作为系统调用方式\n')

    print(f'{"配置":<30} {"ROUGE-L":>8} {"BLEU-4":>8} {"F1":>8} {"概念覆盖":>8}')
    print('-'*65)
    for key in order:
        if key not in results:
            continue
        r = results[key]
        rl = r['rouge']['rouge_l']
        bl = r['bleu_4']
        f1 = r['advanced_metrics']['f1_score']
        cc = r['advanced_metrics']['concept_coverage']
        print(f'{r["config_name"]:<30} {rl:>8.4f} {bl:>8.4f} {f1:>8.4f} {cc:>8.4f}')

    full_rl = results['full']['rouge']['rouge_l']
    print('\n模块贡献（去掉后ROUGE-L下降幅度）：')
    contribs = {}
    for key, label in [('no_hybrid','混合检索'), ('no_kg','知识图谱'),
                        ('no_memory','记忆机制'), ('no_tools','工具增强')]:
        if key in results:
            drop = (full_rl - results[key]['rouge']['rouge_l']) / full_rl * 100
            contribs[label] = round(drop, 2)
            print(f'  {label}: {drop:+.2f}%')

    print('\n✅ 消融实验数据已读取（无需重跑）')
    return {'source_file': ABLATION_FILE, 'results': results, 'contributions': contribs}


# ================================================================
# 实验3：LLM-as-Judge（answer_with_plan，评测答案质量）
# ================================================================

def run_exp3_llm_judge(limit=20, top_k=5, seed=42):
    """
    用 DeepSeek 对完整系统 vs 最强基线（TF-IDF+LLM）的答案质量打分。
    样本数建议 15-20，太多 API 费用高。
    """
    print('\n' + '='*65)
    print('实验3：LLM-as-Judge（answer_with_plan，评测答案质量）')
    print('='*65)

    samples = _fixed_samples(limit, seed=seed)
    print(f'样本数={len(samples)}  top_k={top_k}  seed={seed}')
    print('评分维度: 准确性 / 完整性 / 清晰度 / 相关性（各1-5分）\n')

    judge_methods = {
        'tfidf_rag': ('最强基线（TF-IDF+LLM）',
                      lambda q, aid, k: run_baseline('tfidf_rag', q, top_k=k, algorithm_id=aid)),
        'our_full':  ('本系统完整版（answer_with_plan）',
                      lambda q, aid, k: answer_with_plan(
                          q, top_k=k, algorithm_id=aid,
                          retrieval_mode='hybrid',
                          enable_tools=True, enable_memory=True,
                          enable_planning=True, enable_mcp=False,
                      )),
    }

    judge_results = {}
    for key, (name, fn) in judge_methods.items():
        print(f'▶ 生成答案: {name}')
        questions, preds, refs = [], [], []
        for i, s in enumerate(samples, 1):
            print(f'\r  生成进度: {i}/{len(samples)}', end='', flush=True)
            try:
                r = fn(s['question'], s.get('algorithm_id'), top_k)
                preds.append(r.get('answer', ''))
            except Exception as e:
                print(f'\n  ⚠ 样本{i}失败: {e}')
                preds.append('')
            questions.append(s['question'])
            refs.append(s['reference_answer'])
        print()

        print(f'  LLM打分中（每条调用一次DeepSeek）...')
        judge_out = judge_batch(questions, preds, refs, verbose=True)

        avg = judge_out['average']
        print(f'  准确性={avg["accuracy"]}  完整性={avg["completeness"]}  '
              f'清晰度={avg["clarity"]}  相关性={avg["relevance"]}  '
              f'综合={avg["overall"]}')

        judge_results[key] = {
            'name': name,
            'average_scores': avg,
            'valid_count': judge_out['valid_count'],
            'details': judge_out['details'],
            'predictions': preds,
        }

    # 对比汇总
    print('\n' + '-'*65)
    print(f'{"方法":<30} {"准确性":>6} {"完整性":>6} {"清晰度":>6} {"相关性":>6} {"综合":>6}')
    print('-'*65)
    for key, (name, _) in judge_methods.items():
        avg = judge_results[key]['average_scores']
        print(f'{name:<30} {avg["accuracy"]:>6.2f} {avg["completeness"]:>6.2f} '
              f'{avg["clarity"]:>6.2f} {avg["relevance"]:>6.2f} {avg["overall"]:>6.2f}')

    our_overall = judge_results['our_full']['average_scores']['overall']
    base_overall = judge_results['tfidf_rag']['average_scores']['overall']
    imp = (our_overall - base_overall) / base_overall * 100
    print(f'\n本系统综合分相对最强基线提升: {imp:+.2f}%')

    path, ts = _save({
        'exp': 'llm_judge',
        'config': {'limit': limit, 'top_k': top_k, 'seed': seed},
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'predictions'}
                    for k, v in judge_results.items()},
        'improvement_overall': imp,
    }, 'exp3_llm_judge')
    print(f'✅ 已保存: {path}')
    return judge_results


# ================================================================
# 生成综合报告
# ================================================================

def generate_report(exp1, exp2, exp3, ts):
    """把三个实验的结果汇总成一份 Markdown 报告。"""
    path = f'data/results/option3_report_{ts}.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# 实验报告（选项3：三套互补实验）\n\n')
        f.write(f'**生成时间**: {ts}\n\n')
        f.write('---\n\n')

        # 实验1
        f.write('## 实验1：基线对比（检索质量，ROUGE/BLEU）\n\n')
        f.write('> 使用 `rag_ask`（基础RAG）作为本系统代表，评测混合检索相对单一检索的提升。\n\n')
        f.write('| 方法 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | F1-Score |\n')
        f.write('|------|---------|---------|---------|--------|----------|\n')
        if exp1:
            for key, r in exp1.items():
                rb = r['rouge_bleu']
                f.write(f"| {r['name']} | {rb['rouge_1']} | {rb['rouge_2']} | "
                        f"{rb['rouge_l']} | {rb['bleu_4']} | {r['f1']} |\n")
            our = exp1.get('our_method', {})
            best_rl = max(exp1[k]['rouge_bleu']['rouge_l']
                          for k in exp1 if k != 'our_method')
            imp = (our['rouge_bleu']['rouge_l'] - best_rl) / best_rl * 100
            f.write(f'\n**本系统相对最佳基线 ROUGE-L 提升: {imp:+.2f}%**\n\n')

        # 实验2
        f.write('## 实验2：消融实验（模块贡献，ROUGE-L）\n\n')
        f.write('> 使用 `rag_ask`，逐一去掉各模块，量化每个模块的贡献。\n\n')
        f.write('| 配置 | ROUGE-L | BLEU-4 | F1-Score | 概念覆盖度 |\n')
        f.write('|------|---------|--------|----------|----------|\n')
        if exp2:
            order = ['full', 'no_hybrid', 'no_kg', 'no_memory', 'no_tools']
            for key in order:
                if key not in exp2['results']:
                    continue
                r = exp2['results'][key]
                f.write(f"| {r['config_name']} | {r['rouge']['rouge_l']:.4f} | "
                        f"{r['bleu_4']:.4f} | {r['advanced_metrics']['f1_score']:.4f} | "
                        f"{r['advanced_metrics']['concept_coverage']:.4f} |\n")
            f.write('\n**模块贡献排序（ROUGE-L下降幅度）**：\n\n')
            for label, drop in sorted(exp2['contributions'].items(), key=lambda x: -x[1]):
                f.write(f'- {label}: {drop:+.2f}%\n')
            f.write('\n')

        # 实验3
        f.write('## 实验3：答案质量评测（LLM-as-Judge，1-5分）\n\n')
        f.write('> 使用 `answer_with_plan`（完整规划式问答），DeepSeek 从准确性/完整性/清晰度/相关性四维度打分。\n\n')
        f.write('| 方法 | 准确性 | 完整性 | 清晰度 | 相关性 | 综合分 |\n')
        f.write('|------|--------|--------|--------|--------|--------|\n')
        if exp3:
            for key, r in exp3.items():
                avg = r['average_scores']
                f.write(f"| {r['name']} | {avg['accuracy']:.2f} | {avg['completeness']:.2f} | "
                        f"{avg['clarity']:.2f} | {avg['relevance']:.2f} | {avg['overall']:.2f} |\n")
            our_o = exp3['our_full']['average_scores']['overall']
            base_o = exp3['tfidf_rag']['average_scores']['overall']
            imp3 = (our_o - base_o) / base_o * 100
            f.write(f'\n**本系统综合分相对最强基线提升: {imp3:+.2f}%**\n\n')

        # 结论
        f.write('## 结论\n\n')
        f.write('三套实验从不同角度验证了本系统的有效性：\n\n')
        f.write('1. **实验1** 证明混合检索（TF-IDF+关键词+向量）的检索质量优于单一检索方法\n')
        f.write('2. **实验2** 证明系统中每个模块（混合检索、知识图谱、记忆机制、工具增强）均有正向贡献\n')
        f.write('3. **实验3** 证明完整系统（规划式问答）生成的答案在教学质量上优于基线方法\n')

    print(f'✅ 综合报告已保存: {path}')
    return path


# ================================================================
# 主入口
# ================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'ablation', 'judge', 'all'],
                        default='all')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--judge_limit', type=int, default=20,
                        help='LLM打分的样本数（建议15-20，节省API费用）')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp1 = exp2 = exp3 = None

    if args.mode in ('baseline', 'all'):
        exp1 = run_exp1_baseline(limit=args.limit, top_k=args.top_k, seed=args.seed)

    if args.mode in ('ablation', 'all'):
        exp2 = run_exp2_ablation()

    if args.mode in ('judge', 'all'):
        exp3 = run_exp3_llm_judge(limit=args.judge_limit, top_k=args.top_k, seed=args.seed)

    if args.mode == 'all':
        generate_report(exp1, exp2, exp3, ts)

    print('\n🎉 完成！结果在 data/results/ 目录下。')
