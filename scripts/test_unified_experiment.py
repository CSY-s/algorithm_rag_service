"""测试统一实验脚本 - 小规模验证"""

import sys
sys.path.insert(0, '.')

from app.db_source import fetch_algorithms
from app.rag import rag_ask
from app.baselines import bm25_baseline, llm_only_baseline
from app.metrics import score_answer

# 测试1: 检查数据获取
print("="*60)
print("测试1: 检查数据获取")
print("="*60)

algos = fetch_algorithms()
print(f"算法总数: {len(algos)}")

# 找出有关联的算法组
algo_groups = {
    '顺序表': [a for a in algos if '顺序表' in a['algorithm_name']],
    '单链表': [a for a in algos if '单链表' in a['algorithm_name']],
}

for group_name, group_algos in algo_groups.items():
    print(f"\n{group_name}组: {len(group_algos)}个算法")
    for a in group_algos[:3]:
        print(f"  - {a['algorithm_name']}")

# 测试2: 生成中等难度问题
print("\n" + "="*60)
print("测试2: 生成中等难度问题")
print("="*60)

if len(algo_groups['顺序表']) >= 2:
    a1, a2 = algo_groups['顺序表'][0], algo_groups['顺序表'][1]
    question = f'{a1["algorithm_name"]}和{a2["algorithm_name"]}有什么区别？'
    print(f"\n问题: {question}")
    
    # 测试3: 调用基线方法
    print("\n" + "="*60)
    print("测试3: 调用基线方法")
    print("="*60)
    
    print("\n[1] BM25基线...")
    try:
        result = bm25_baseline(question)
        answer = result['answer'] if isinstance(result, dict) else result
        print(f"答案长度: {len(answer)} 字符")
        print(f"答案预览: {answer[:100]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n[2] 纯LLM基线...")
    try:
        result = llm_only_baseline(question)
        answer = result['answer'] if isinstance(result, dict) else result
        print(f"答案长度: {len(answer)} 字符")
        print(f"答案预览: {answer[:100]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n[3] 本系统...")
    try:
        result = rag_ask(question)
        answer = result['answer'] if isinstance(result, dict) else result
        print(f"答案长度: {len(answer)} 字符")
        print(f"答案预览: {answer[:100]}...")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试4: 计算指标
    print("\n" + "="*60)
    print("测试4: 计算指标")
    print("="*60)
    
    reference = f'{a1["algorithm_name"]}的步骤是:{a1["step_text"][:100]}... {a2["algorithm_name"]}的步骤是:{a2["step_text"][:100]}...'
    
    try:
        scores = score_answer(answer, reference)
        print(f"ROUGE-L: {scores['rouge_l']:.4f}")
        print(f"BLEU-4: {scores['bleu_4']:.4f}")
    except Exception as e:
        print(f"错误: {e}")

print("\n" + "="*60)
print("测试完成!")
print("="*60)
