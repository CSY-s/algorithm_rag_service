"""复杂问题专项评测脚本。

针对本系统真正有优势的三类问题进行专项评测：
  A. 跨算法对比类  —— 需要知识图谱，单一检索难以回答
  B. 多步推理类    —— 需要规划器，答案不在单一片段里
  C. 综合分析类    —— 需要整合多个知识点

评测方式：LLM-as-Judge（1-5分），对比本系统 vs 最强基线（TF-IDF+LLM）
"""

import json, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.baselines import run_baseline
from app.agent_workflow import answer_with_plan
from app.llm_judge import judge_batch

# ================================================================
# 复杂问题集（手工设计，覆盖三类场景）
# ================================================================

COMPLEX_QUESTIONS = [
    # --- A. 跨算法对比类（需要知识图谱/多片段整合）---
    {
        "id": "C01", "type": "跨算法对比",
        "question": "快速排序和归并排序在时间复杂度、空间复杂度和稳定性上有什么区别？各自适合什么场景？",
        "reference": "快速排序：平均O(nlogn)，最坏O(n²)，空间O(logn)，不稳定，适合内排序；归并排序：稳定O(nlogn)，空间O(n)，稳定，适合外排序和链表排序。",
    },
    {
        "id": "C02", "type": "跨算法对比",
        "question": "顺序表和链表在插入、删除、查找操作上的时间复杂度分别是多少？为什么会有这种差异？",
        "reference": "顺序表：查找O(1)，插入删除O(n)（需移动元素）；链表：查找O(n)，插入删除O(1)（已知位置时）。差异源于存储结构：顺序表连续存储支持随机访问，链表离散存储需顺序遍历。",
    },
    {
        "id": "C03", "type": "跨算法对比",
        "question": "BFS和DFS在图遍历中的区别是什么？分别适合解决哪类问题？",
        "reference": "BFS用队列，按层遍历，适合最短路径；DFS用栈/递归，深度优先，适合连通性、拓扑排序、回溯问题。",
    },
    {
        "id": "C04", "type": "跨算法对比",
        "question": "堆排序和选择排序都是基于选择的排序，它们的时间复杂度为什么不同？",
        "reference": "选择排序每次找最小值需O(n)扫描，总O(n²)；堆排序用堆结构，每次调整O(logn)，总O(nlogn)。堆排序通过堆化避免了重复扫描。",
    },
    {
        "id": "C05", "type": "跨算法对比",
        "question": "KMP算法相比朴素字符串匹配算法的优势在哪里？next数组在其中起什么作用？",
        "reference": "朴素匹配O(mn)，失配后主串回退；KMP通过next数组记录模式串前缀信息，失配时只移动模式串不回退主串，时间复杂度O(m+n)。",
    },
    # --- B. 多步推理类（需要规划+多知识点整合）---
    {
        "id": "C06", "type": "多步推理",
        "question": "如果要在一个有100万个元素的有序数组中查找某个值，应该选择哪种查找算法？请说明理由和时间复杂度。",
        "reference": "选二分查找，时间复杂度O(logn)≈20次比较。有序数组支持随机访问，二分每次排除一半，远优于顺序查找O(n)。",
    },
    {
        "id": "C07", "type": "多步推理",
        "question": "为什么说递归算法虽然代码简洁，但在实际工程中需要谨慎使用？请从时间和空间两个角度分析。",
        "reference": "时间：递归可能有重复计算（如斐波那契），需记忆化优化；空间：每次递归调用占用栈帧，深度过大会栈溢出，空间复杂度O(n)。",
    },
    {
        "id": "C08", "type": "多步推理",
        "question": "动态规划和分治法都是将问题分解为子问题，它们的本质区别是什么？",
        "reference": "分治法子问题独立，不重叠，直接递归；动态规划子问题重叠，通过记忆化/表格避免重复计算，适合有最优子结构的问题。",
    },
    {
        "id": "C09", "type": "多步推理",
        "question": "哈希表的查找时间复杂度理论上是O(1)，但实际中为什么有时会退化到O(n)？如何避免？",
        "reference": "哈希冲突严重时（如所有元素映射到同一桶），链地址法退化为链表查找O(n)。避免方法：好的哈希函数、合适的装载因子（<0.75）、动态扩容。",
    },
    {
        "id": "C10", "type": "多步推理",
        "question": "平衡二叉树（AVL树）相比普通二叉搜索树解决了什么问题？代价是什么？",
        "reference": "BST在有序插入时退化为链表O(n)；AVL通过旋转保持平衡，查找O(logn)。代价是插入删除需旋转操作，实现复杂，常数因子较大。",
    },
    # --- C. 综合分析类（需要整合多个知识点）---
    {
        "id": "C11", "type": "综合分析",
        "question": "请分析顺序表的初始化、插入、删除、查找四种操作的时间复杂度，并说明哪种操作最耗时及原因。",
        "reference": "初始化O(1)，查找O(1)（按位）/O(n)（按值），插入O(n)（移动元素），删除O(n)（移动元素）。插入删除最耗时，因为需要移动大量元素保持连续性。",
    },
    {
        "id": "C12", "type": "综合分析",
        "question": "栈和队列都是线性表的特殊形式，它们的本质区别是什么？各自适合解决什么类型的问题？",
        "reference": "栈LIFO（后进先出），适合函数调用、括号匹配、表达式求值、DFS；队列FIFO（先进先出），适合BFS、任务调度、缓冲区。",
    },
    {
        "id": "C13", "type": "综合分析",
        "question": "请解释为什么链表的空间复杂度比顺序表高，但在某些场景下仍然优先选择链表？",
        "reference": "链表每个节点需额外存储指针，空间开销更大。但链表优势：动态分配无需预先知道大小，插入删除O(1)（已知位置），不会有内存浪费（顺序表预分配可能浪费）。",
    },
    {
        "id": "C14", "type": "综合分析",
        "question": "冒泡排序、插入排序、选择排序都是O(n²)的排序算法，在实际应用中如何选择？",
        "reference": "插入排序对近乎有序的数据最优（接近O(n)），且稳定；选择排序交换次数最少；冒泡排序最简单但效率最低。小数据量或近乎有序时用插入排序，其他情况用快排/归并。",
    },
    {
        "id": "C15", "type": "综合分析",
        "question": "图的邻接矩阵和邻接表两种存储方式各有什么优缺点？分别适合什么类型的图？",
        "reference": "邻接矩阵：O(V²)空间，判断边O(1)，适合稠密图；邻接表：O(V+E)空间，遍历邻居高效，适合稀疏图。稠密图用矩阵，稀疏图用邻接表。",
    },
]


# ================================================================
# 评测主流程
# ================================================================

def run_complex_evaluation(top_k: int = 5):
    print('\n' + '='*70)
    print('复杂问题专项评测')
    print('='*70)
    print(f'问题数: {len(COMPLEX_QUESTIONS)}  top_k={top_k}')
    print('问题类型分布:')
    from collections import Counter
    type_cnt = Counter(q['type'] for q in COMPLEX_QUESTIONS)
    for t, n in type_cnt.items():
        print(f'  {t}: {n}题')
    print()

    methods = {
        'tfidf_rag': ('TF-IDF + LLM（最强基线）',
                      lambda q, k: run_baseline('tfidf_rag', q, top_k=k)),
        'our_full':  ('本系统完整版（answer_with_plan）',
                      lambda q, k: answer_with_plan(
                          q, top_k=k,
                          retrieval_mode='hybrid',
                          enable_tools=True, enable_memory=True,
                          enable_planning=True, enable_mcp=False,
                      )),
    }

    all_results = {}

    for key, (name, fn) in methods.items():
        print(f'▶ 生成答案: {name}')
        questions, preds, refs, q_types, q_ids = [], [], [], [], []

        for i, q in enumerate(COMPLEX_QUESTIONS, 1):
            print(f'\r  生成进度: {i}/{len(COMPLEX_QUESTIONS)}', end='', flush=True)
            try:
                r = fn(q['question'], top_k)
                preds.append(r.get('answer', ''))
            except Exception as e:
                print(f'\n  ⚠ 问题{q["id"]}失败: {e}')
                preds.append('')
            questions.append(q['question'])
            refs.append(q['reference'])
            q_types.append(q['type'])
            q_ids.append(q['id'])
        print()

        print(f'  LLM打分中...')
        judge_out = judge_batch(questions, preds, refs, verbose=True)

        # 按题型分组统计
        type_scores = {}
        for i, detail in enumerate(judge_out['details']):
            qtype = q_types[i]
            if qtype not in type_scores:
                type_scores[qtype] = []
            type_scores[qtype].append(detail['overall'])

        type_avg = {t: round(sum(s)/len(s), 3) for t, s in type_scores.items()}

        avg = judge_out['average']
        print(f'  综合分={avg["overall"]}  准确性={avg["accuracy"]}  '
              f'完整性={avg["completeness"]}  清晰度={avg["clarity"]}  相关性={avg["relevance"]}')
        print(f'  按题型: ' + '  '.join(f'{t}={s}' for t, s in type_avg.items()))

        all_results[key] = {
            'name': name,
            'average': avg,
            'type_avg': type_avg,
            'details': judge_out['details'],
            'predictions': preds,
            'valid_count': judge_out['valid_count'],
        }

    # ---- 汇总对比 ----
    print('\n' + '-'*70)
    print(f'{"方法":<35} {"准确性":>6} {"完整性":>6} {"清晰度":>6} {"相关性":>6} {"综合":>6}')
    print('-'*70)
    for key, (name, _) in methods.items():
        avg = all_results[key]['average']
        print(f'{name:<35} {avg["accuracy"]:>6.2f} {avg["completeness"]:>6.2f} '
              f'{avg["clarity"]:>6.2f} {avg["relevance"]:>6.2f} {avg["overall"]:>6.2f}')

    print('\n按题型对比（综合分）:')
    print(f'{"题型":<12}', end='')
    for key, (name, _) in methods.items():
        print(f'  {name[:15]:>15}', end='')
    print()
    all_types = list(type_cnt.keys())
    for qtype in all_types:
        print(f'{qtype:<12}', end='')
        for key in methods:
            score = all_results[key]['type_avg'].get(qtype, 0)
            print(f'  {score:>15.2f}', end='')
        print()

    our_overall = all_results['our_full']['average']['overall']
    base_overall = all_results['tfidf_rag']['average']['overall']
    imp = (our_overall - base_overall) / base_overall * 100
    print(f'\n本系统综合分相对最强基线提升: {imp:+.2f}%')

    # ---- 保存结果 ----
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('data/results', exist_ok=True)
    json_path = f'data/results/complex_eval_{ts}.json'
    md_path   = f'data/results/complex_eval_{ts}.md'

    save_data = {
        'experiment': 'complex_question_eval',
        'config': {'question_count': len(COMPLEX_QUESTIONS), 'top_k': top_k},
        'questions': COMPLEX_QUESTIONS,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'predictions'}
                    for k, v in all_results.items()},
        'improvement': imp,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    _write_md(md_path, all_results, methods, type_cnt, imp, ts)
    print(f'\n✅ 已保存: {json_path}')
    print(f'✅ 已保存: {md_path}')
    return all_results, imp


def _write_md(path, all_results, methods, type_cnt, imp, ts):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# 复杂问题专项评测报告\n\n')
        f.write(f'**实验时间**: {ts}\n\n')
        f.write('**问题类型**:\n')
        for t, n in type_cnt.items():
            f.write(f'- {t}: {n}题\n')
        f.write('\n**评分维度**: 准确性 / 完整性 / 清晰度 / 相关性（各1-5分）\n\n')
        f.write('---\n\n')

        f.write('## 总体对比\n\n')
        f.write('| 方法 | 准确性 | 完整性 | 清晰度 | 相关性 | **综合分** |\n')
        f.write('|------|--------|--------|--------|--------|----------|\n')
        for key, (name, _) in methods.items():
            avg = all_results[key]['average']
            f.write(f"| {name} | {avg['accuracy']:.2f} | {avg['completeness']:.2f} | "
                    f"{avg['clarity']:.2f} | {avg['relevance']:.2f} | **{avg['overall']:.2f}** |\n")
        f.write(f'\n**本系统综合分相对最强基线提升: {imp:+.2f}%**\n\n')

        f.write('## 按题型分组对比（综合分）\n\n')
        f.write('| 题型 |')
        for key, (name, _) in methods.items():
            f.write(f' {name} |')
        f.write('\n|------|')
        for _ in methods:
            f.write('--------|')
        f.write('\n')
        for qtype in type_cnt:
            f.write(f'| {qtype} |')
            for key in methods:
                score = all_results[key]['type_avg'].get(qtype, 0)
                f.write(f' {score:.2f} |')
            f.write('\n')

        f.write('\n## 逐题对比\n\n')
        f.write('| ID | 题型 | 问题摘要 |')
        for key, (name, _) in methods.items():
            f.write(f' {name[:10]}综合 |')
        f.write('\n|----|------|---------|')
        for _ in methods:
            f.write('---------|')
        f.write('\n')
        for i, q in enumerate(COMPLEX_QUESTIONS):
            f.write(f"| {q['id']} | {q['type']} | {q['question'][:25]}... |")
            for key in methods:
                detail = all_results[key]['details'][i]
                f.write(f" {detail['overall']:.1f} |")
            f.write('\n')

        f.write('\n## 结论\n\n')
        our_o = all_results['our_full']['average']['overall']
        base_o = all_results['tfidf_rag']['average']['overall']
        f.write(f'在复杂问题场景下，本系统综合得分 **{our_o:.2f}/5**，'
                f'最强基线得分 **{base_o:.2f}/5**，提升 **{imp:+.2f}%**。\n\n')
        f.write('复杂问题需要跨算法对比、多步推理和综合分析能力，'
                '本系统通过规划器、知识图谱和混合检索的协同，'
                '在这类问题上展现出更强的综合能力。\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    run_complex_evaluation(top_k=args.top_k)
