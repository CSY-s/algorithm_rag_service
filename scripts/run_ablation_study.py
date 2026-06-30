"""运行消融实验（Ablation Study）。

系统地移除各个模块，评估每个模块的贡献。"""

import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import rag_ask
from app.db_source import fetch_eval_samples
from app.metrics import score_answer
from app.advanced_metrics import batch_evaluate


def run_ablation_study(
    limit: int = 30,
    algorithm_id: int | None = None,
    top_k: int = 5,
):
    """运行完整的消融实验。
    
    测试配置：
    1. Full: 完整系统（混合检索 + 知识图谱 + 记忆 + 扩展）
    2. -Hybrid: 去掉混合检索，只用 TF-IDF
    3. -KG: 去掉知识图谱增强
    4. -Memory: 去掉记忆机制
    5. -Tools: 去掉工具增强（知识图谱工具）
    
    Args:
        limit: 测试样本数量
        algorithm_id: 算法 ID（可选）
        top_k: 检索数量
    """
    print("=" * 80)
    print("消融实验（Ablation Study）")
    print("=" * 80)
    print(f"测试样本数: {limit}")
    print(f"检索数量 top_k: {top_k}")
    print(f"算法 ID: {algorithm_id or '全部'}")
    print()
    
    # 1. 获取测试样本
    print("📊 加载测试样本...")
    samples = fetch_eval_samples(limit=limit, algorithm_id=algorithm_id)
    print(f"✅ 加载了 {len(samples)} 个测试样本\n")
    
    if len(samples) == 0:
        print("❌ 没有测试样本，退出")
        return
    
    # 2. 定义消融配置
    configs = {
        'full': {
            'name': '完整系统',
            'retrieval_mode': 'hybrid',
            'enable_tools': True,
            'enable_memory': True,
        },
        'no_hybrid': {
            'name': '去掉混合检索（只用 TF-IDF）',
            'retrieval_mode': 'tfidf',
            'enable_tools': True,
            'enable_memory': True,
        },
        'no_kg': {
            'name': '去掉知识图谱增强',
            'retrieval_mode': 'hybrid',
            'enable_tools': False,  # 工具主要是知识图谱查询
            'enable_memory': True,
        },
        'no_memory': {
            'name': '去掉记忆机制',
            'retrieval_mode': 'hybrid',
            'enable_tools': True,
            'enable_memory': False,
        },
        'no_tools': {
            'name': '去掉工具增强',
            'retrieval_mode': 'hybrid',
            'enable_tools': False,
            'enable_memory': True,
        },
    }
    
    # 3. 运行实验
    results = {}
    
    for config_key, config in configs.items():
        print(f"\n{'='*80}")
        print(f"运行配置: {config['name']}")
        print(f"{'='*80}")
        print(f"  检索模式: {config['retrieval_mode']}")
        print(f"  工具增强: {config['enable_tools']}")
        print(f"  记忆机制: {config['enable_memory']}")
        print()
        
        predictions = []
        references = []
        config_results = []
        
        for i, sample in enumerate(samples, 1):
            question = sample['question']
            reference = sample['reference_answer']
            algo_id = sample.get('algorithm_id')
            
            print(f"\r进度: {i}/{len(samples)}", end='', flush=True)
            
            try:
                # 运行配置
                result = rag_ask(
                    question,
                    top_k=top_k,
                    algorithm_id=algo_id,
                    retrieval_mode=config['retrieval_mode'],
                    enable_tools=config['enable_tools'],
                    enable_memory=config['enable_memory'],
                )
                
                prediction = result['answer']
                predictions.append(prediction)
                references.append(reference)
                
                config_results.append({
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
                config_results.append({
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
        results[config_key] = {
            'config_name': config['name'],
            'config': config,
            'rouge': avg_rouge,
            'bleu_4': avg_bleu,
            'advanced_metrics': advanced_metrics['average_metrics'],
            'sample_count': len(samples),
            'detailed_results': config_results,
        }
        
        # 打印结果
        print(f"\n✅ {config['name']} 结果:")
        print(f"   ROUGE-1: {avg_rouge['rouge_1']:.4f}")
        print(f"   ROUGE-2: {avg_rouge['rouge_2']:.4f}")
        print(f"   ROUGE-L: {avg_rouge['rouge_l']:.4f}")
        print(f"   BLEU-4:  {avg_bleu:.4f}")
        print(f"   F1-Score: {advanced_metrics['average_metrics']['f1_score']:.4f}")
        print(f"   概念覆盖度: {advanced_metrics['average_metrics']['concept_coverage']:.4f}")
    
    # 5. 生成对比表格
    print(f"\n{'='*80}")
    print("消融实验结果汇总")
    print(f"{'='*80}\n")
    
    print(f"{'配置':<35} {'ROUGE-L':<10} {'BLEU-4':<10} {'F1':<10} {'概念覆盖':<10}")
    print("-" * 75)
    
    for config_key in ['full', 'no_hybrid', 'no_kg', 'no_memory', 'no_tools']:
        result = results[config_key]
        config_name = result['config_name']
        rouge_l = result['rouge']['rouge_l']
        bleu_4 = result['bleu_4']
        f1 = result['advanced_metrics']['f1_score']
        concept = result['advanced_metrics']['concept_coverage']
        
        print(f"{config_name:<35} {rouge_l:<10.4f} {bleu_4:<10.4f} {f1:<10.4f} {concept:<10.4f}")
    
    # 6. 计算各模块的贡献
    print(f"\n{'='*80}")
    print("各模块的贡献分析")
    print(f"{'='*80}\n")
    
    full_result = results['full']
    
    # 混合检索的贡献
    hybrid_contribution = (
        full_result['rouge']['rouge_l'] - results['no_hybrid']['rouge']['rouge_l']
    ) / full_result['rouge']['rouge_l'] * 100
    
    # 知识图谱的贡献
    kg_contribution = (
        full_result['rouge']['rouge_l'] - results['no_kg']['rouge']['rouge_l']
    ) / full_result['rouge']['rouge_l'] * 100
    
    # 记忆机制的贡献
    memory_contribution = (
        full_result['rouge']['rouge_l'] - results['no_memory']['rouge']['rouge_l']
    ) / full_result['rouge']['rouge_l'] * 100
    
    # 工具增强的贡献
    tools_contribution = (
        full_result['rouge']['rouge_l'] - results['no_tools']['rouge']['rouge_l']
    ) / full_result['rouge']['rouge_l'] * 100
    
    print(f"混合检索贡献: {hybrid_contribution:.2f}% (去掉后性能下降)")
    print(f"知识图谱贡献: {kg_contribution:.2f}% (去掉后性能下降)")
    print(f"记忆机制贡献: {memory_contribution:.2f}% (去掉后性能下降)")
    print(f"工具增强贡献: {tools_contribution:.2f}% (去掉后性能下降)")
    
    # 7. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'data/results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'ablation_study_{timestamp}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'experiment_config': {
                    'limit': limit,
                    'algorithm_id': algorithm_id,
                    'top_k': top_k,
                    'timestamp': timestamp,
                },
                'results': results,
                'contributions': {
                    'hybrid_retrieval': hybrid_contribution,
                    'knowledge_graph': kg_contribution,
                    'memory': memory_contribution,
                    'tools': tools_contribution,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    # 8. 生成 Markdown 报告
    md_file = os.path.join(output_dir, f'ablation_study_{timestamp}.md')
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# 消融实验报告\n\n")
        f.write(f"**实验时间**: {timestamp}\n\n")
        f.write(f"**实验配置**:\n")
        f.write(f"- 测试样本数: {limit}\n")
        f.write(f"- 检索数量 top_k: {top_k}\n")
        f.write(f"- 算法 ID: {algorithm_id or '全部'}\n\n")
        
        f.write(f"## 实验结果\n\n")
        f.write(f"| 配置 | ROUGE-L | BLEU-4 | F1-Score | 概念覆盖度 |\n")
        f.write(f"|------|---------|--------|----------|----------|\n")
        
        for config_key in ['full', 'no_hybrid', 'no_kg', 'no_memory', 'no_tools']:
            result = results[config_key]
            config_name = result['config_name']
            rouge_l = result['rouge']['rouge_l']
            bleu_4 = result['bleu_4']
            f1 = result['advanced_metrics']['f1_score']
            concept = result['advanced_metrics']['concept_coverage']
            
            f.write(f"| {config_name} | {rouge_l:.4f} | {bleu_4:.4f} | {f1:.4f} | {concept:.4f} |\n")
        
        f.write(f"\n## 模块贡献分析\n\n")
        f.write(f"各模块对系统性能的贡献（去掉后性能下降幅度）：\n\n")
        f.write(f"- **混合检索**: {hybrid_contribution:.2f}%\n")
        f.write(f"- **知识图谱**: {kg_contribution:.2f}%\n")
        f.write(f"- **记忆机制**: {memory_contribution:.2f}%\n")
        f.write(f"- **工具增强**: {tools_contribution:.2f}%\n")
        
        f.write(f"\n## 结论\n\n")
        
        # 找出贡献最大的模块
        contributions = {
            '混合检索': hybrid_contribution,
            '知识图谱': kg_contribution,
            '记忆机制': memory_contribution,
            '工具增强': tools_contribution,
        }
        
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        f.write(f"模块重要性排序：\n\n")
        for i, (module, contrib) in enumerate(sorted_contributions, 1):
            f.write(f"{i}. **{module}**: {contrib:.2f}%\n")
    
    print(f"✅ Markdown 报告已保存到: {md_file}")
    
    return results


if __name__ == '__main__':
    # 运行消融实验
    run_ablation_study(
        limit=30,  # 测试 30 个样本
        algorithm_id=None,  # 全部算法
        top_k=5,  # 检索 top 5
    )
