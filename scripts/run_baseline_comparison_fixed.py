"""运行基线方法对比实验（修复版）。

确保使用与消融实验相同的测试样本。"""

import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.baselines import run_baseline
from app.rag import rag_ask
from app.db_source import fetch_eval_samples
from app.metrics import score_answer
from app.advanced_metrics import batch_evaluate


def run_comparison_experiment_fixed(
    limit: int = 30,
    algorithm_id: int | None = None,
    top_k: int = 5,
    seed: int = 42,
):
    """运行基线对比实验（使用固定种子确保可复现）。
    
    Args:
        limit: 测试样本数量
        algorithm_id: 算法 ID（可选）
        top_k: 检索数量
        seed: 随机种子
    """
    print("=" * 80)
    print("基线方法对比实验（修复版）")
    print("=" * 80)
    print(f"测试样本数: {limit}")
    print(f"检索数量 top_k: {top_k}")
    print(f"随机种子: {seed}")
    print(f"算法 ID: {algorithm_id or '全部'}")
    print()
    
    # 1. 获取测试样本（使用固定种子）
    print("📊 加载测试样本...")
    import random
    random.seed(seed)
    
    samples = fetch_eval_samples(limit=limit, algorithm_id=algorithm_id)
    
    # 打乱顺序（使用固定种子）
    random.shuffle(samples)
    
    print(f"✅ 加载了 {len(samples)} 个测试样本")
    print(f"   第一个样本ID: {samples[0].get('sample_id')}")
    print(f"   最后一个样本ID: {samples[-1].get('sample_id')}\n")
    
    if len(samples) == 0:
        print("❌ 没有测试样本，退出")
        return
    
    # 2. 定义要对比的方法
    methods = {
        'bm25': 'BM25 检索 + 抽取式回答',
        'llm_only': '纯 LLM 生成（无检索）',
        'vector_rag': '标准 RAG（向量检索）',
        'tfidf_rag': '单一 TF-IDF 检索 + LLM',
        'our_method': '本文方法（混合检索 + 知识图谱 + 记忆）',
    }
    
    # 3. 运行实验
    results = {}
    
    for method_key, method_name in methods.items():
        print(f"\n{'='*80}")
        print(f"运行方法: {method_name}")
        print(f"{'='*80}")
        
        predictions = []
        references = []
        method_results = []
        
        for i, sample in enumerate(samples, 1):
            question = sample['question']
            reference = sample['reference_answer']
            algo_id = sample.get('algorithm_id')
            
            print(f"\r进度: {i}/{len(samples)}", end='', flush=True)
            
            try:
                # 运行方法
                if method_key == 'our_method':
                    # 本文方法
                    result = rag_ask(
                        question,
                        top_k=top_k,
                        algorithm_id=algo_id,
                        retrieval_mode='hybrid',
                        enable_tools=True,
                        enable_memory=True,
                    )
                else:
                    # 基线方法
                    result = run_baseline(
                        method_key,
                        question,
                        top_k=top_k,
                        algorithm_id=algo_id,
                    )
                
                prediction = result['answer']
                predictions.append(prediction)
                references.append(reference)
                
                method_results.append({
                    'sample_id': sample.get('sample_id'),
                    'question': question,
                    'prediction': prediction,
                    'reference': reference,
                    'algorithm_name': sample.get('algorithm_name'),
                })
            
            except Exception as e:
                print(f"\n⚠️  样本 {i} 失败: {str(e)}")
                predictions.append('')
                references.append(reference)
                method_results.append({
                    'sample_id': sample.get('sample_id'),
                    'question': question,
                    'prediction': '',
                    'reference': reference,
                    'error': str(e),
                })
        
        print()  # 换行
        
        # 4. 计算指标
        print("📊 计算评测指标...")
        
        # ROUGE 和 BLEU
        scores = []
        
        for pred, ref in zip(predictions, references):
            if pred:
                score = score_answer(pred, ref)
                scores.append(score)
        
        # 平均 ROUGE 和 BLEU
        avg_rouge = {
            'rouge_1': sum(s['rouge_1'] for s in scores) / len(scores) if scores else 0,
            'rouge_2': sum(s['rouge_2'] for s in scores) / len(scores) if scores else 0,
            'rouge_l': sum(s['rouge_l'] for s in scores) / len(scores) if scores else 0,
        }
        
        avg_bleu = sum(s['bleu_4'] for s in scores) / len(scores) if scores else 0
        
        # 高级指标
        advanced_metrics = batch_evaluate(predictions, references)
        
        # 保存结果
        results[method_key] = {
            'method_name': method_name,
            'rouge': avg_rouge,
            'bleu_4': avg_bleu,
            'advanced_metrics': advanced_metrics['average_metrics'],
            'sample_count': len(samples),
            'detailed_results': method_results,
        }
        
        # 打印结果
        print(f"\n✅ {method_name} 结果:")
        print(f"   ROUGE-1: {avg_rouge['rouge_1']:.4f}")
        print(f"   ROUGE-2: {avg_rouge['rouge_2']:.4f}")
        print(f"   ROUGE-L: {avg_rouge['rouge_l']:.4f}")
        print(f"   BLEU-4:  {avg_bleu:.4f}")
        print(f"   F1-Score: {advanced_metrics['average_metrics']['f1_score']:.4f}")
        print(f"   概念覆盖度: {advanced_metrics['average_metrics']['concept_coverage']:.4f}")
    
    # 5. 生成对比表格
    print(f"\n{'='*80}")
    print("对比结果汇总")
    print(f"{'='*80}\n")
    
    print(f"{'方法':<35} {'ROUGE-L':<10} {'BLEU-4':<10} {'F1':<10} {'概念覆盖':<10}")
    print("-" * 75)
    
    for method_key, method_name in methods.items():
        result = results[method_key]
        rouge_l = result['rouge']['rouge_l']
        bleu_4 = result['bleu_4']
        f1 = result['advanced_metrics']['f1_score']
        concept = result['advanced_metrics']['concept_coverage']
        
        print(f"{method_name:<35} {rouge_l:<10.4f} {bleu_4:<10.4f} {f1:<10.4f} {concept:<10.4f}")
    
    # 6. 计算提升幅度
    print(f"\n{'='*80}")
    print("相对最佳基线的提升幅度")
    print(f"{'='*80}\n")
    
    our_result = results['our_method']
    baseline_results = {k: v for k, v in results.items() if k != 'our_method'}
    
    # 找到每个指标的最佳基线
    best_rouge_l = max(r['rouge']['rouge_l'] for r in baseline_results.values())
    best_bleu_4 = max(r['bleu_4'] for r in baseline_results.values())
    best_f1 = max(r['advanced_metrics']['f1_score'] for r in baseline_results.values())
    
    # 计算提升
    rouge_l_improvement = (our_result['rouge']['rouge_l'] - best_rouge_l) / best_rouge_l * 100
    bleu_4_improvement = (our_result['bleu_4'] - best_bleu_4) / best_bleu_4 * 100
    f1_improvement = (our_result['advanced_metrics']['f1_score'] - best_f1) / best_f1 * 100
    
    print(f"ROUGE-L 提升: {rouge_l_improvement:+.2f}%")
    print(f"BLEU-4 提升:  {bleu_4_improvement:+.2f}%")
    print(f"F1-Score 提升: {f1_improvement:+.2f}%")
    
    # 7. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'data/results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'baseline_comparison_fixed_{timestamp}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'experiment_config': {
                    'limit': limit,
                    'algorithm_id': algorithm_id,
                    'top_k': top_k,
                    'seed': seed,
                    'timestamp': timestamp,
                    'sample_ids': [s.get('sample_id') for s in samples],
                },
                'results': results,
                'improvements': {
                    'rouge_l': rouge_l_improvement,
                    'bleu_4': bleu_4_improvement,
                    'f1_score': f1_improvement,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return results


if __name__ == '__main__':
    # 运行实验（使用固定种子确保可复现）
    run_comparison_experiment_fixed(
        limit=30,
        algorithm_id=None,
        top_k=5,
        seed=42,  # 固定种子
    )
