"""查看知识库现有数据,找出适合构建测试集的素材"""

import sys
sys.path.insert(0, '.')

from app.db_source import fetch_algorithms, fetch_eval_samples

def main():
    # 获取算法数据
    algos = fetch_algorithms()
    print(f"算法总数: {len(algos)}")
    
    print("\n前10个算法:")
    for a in algos[:10]:
        print(f"  {a['algorithm_id']}: {a['algorithm_name']}")
        print(f"    问答数: {len(a.get('question_docs', []))}")
    
    # 获取问答样本
    samples = fetch_eval_samples(limit=100)
    print(f"\n问答样本总数: {len(samples)}")
    
    print("\n前5个问答:")
    for s in samples[:5]:
        print(f"  Q: {s['question'][:60]}")
        print(f"  A: {s['reference_answer'][:100]}\n")
    
    # 统计问答类型
    print("\n问答类型分析:")
    step_count = sum(1 for s in samples if '步骤' in s['question'] or '过程' in s['question'])
    complexity_count = sum(1 for s in samples if '复杂度' in s['question'] or '时间' in s['question'])
    code_count = sum(1 for s in samples if '代码' in s['question'] or '实现' in s['question'])
    
    print(f"  步骤类问题: {step_count}")
    print(f"  复杂度类问题: {complexity_count}")
    print(f"  代码类问题: {code_count}")
    print(f"  其他问题: {len(samples) - step_count - complexity_count - code_count}")

if __name__ == '__main__':
    main()
