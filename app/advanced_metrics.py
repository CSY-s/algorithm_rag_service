"""高级评测指标模块。

补充 BERTScore、概念覆盖度、解释清晰度等教育领域指标。"""

import re
from typing import List, Dict, Optional

import jieba


# ============================================================================
# 1. BERTScore（语义相似度）
# ============================================================================

def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算 BERTScore（基于预训练模型的语义相似度）。
    
    注意：需要安装 bert-score 库：pip install bert-score
    
    Args:
        predictions: 预测答案列表
        references: 参考答案列表
    
    Returns:
        包含 precision、recall、f1 的字典
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(
            predictions,
            references,
            lang='zh',
            verbose=False,
            device='cpu',  # 使用 CPU，避免 GPU 依赖
        )
        
        return {
            'bertscore_precision': float(P.mean().item()),
            'bertscore_recall': float(R.mean().item()),
            'bertscore_f1': float(F1.mean().item()),
        }
    
    except ImportError:
        print("⚠️  警告：未安装 bert-score 库，跳过 BERTScore 计算")
        print("   安装命令：pip install bert-score")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
        }
    
    except Exception as e:
        print(f"⚠️  BERTScore 计算失败: {str(e)}")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
        }


# ============================================================================
# 2. F1-Score（基于词汇重叠）
# ============================================================================

def compute_f1_score(prediction: str, reference: str) -> float:
    """计算 F1-Score（基于词汇重叠）。
    
    Args:
        prediction: 预测答案
        reference: 参考答案
    
    Returns:
        F1 分数（0-1）
    """
    # 分词
    pred_tokens = set(jieba.lcut(prediction))
    ref_tokens = set(jieba.lcut(reference))
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # 计算交集
    common = pred_tokens & ref_tokens
    
    if len(common) == 0:
        return 0.0
    
    # 计算精确率和召回率
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    # 计算 F1
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ============================================================================
# 3. 概念覆盖度（教育领域指标）
# ============================================================================

def extract_algorithm_concepts(text: str) -> List[str]:
    """从文本中提取算法相关概念。
    
    Args:
        text: 输入文本
    
    Returns:
        概念列表
    """
    # 算法领域的核心概念关键词
    concept_keywords = [
        # 时间复杂度
        'O(1)', 'O(n)', 'O(log n)', 'O(n log n)', 'O(n^2)', 'O(2^n)',
        '常数时间', '线性时间', '对数时间', '平方时间', '指数时间',
        '时间复杂度', '空间复杂度',
        
        # 数据结构
        '数组', '链表', '栈', '队列', '树', '图', '哈希表', '堆',
        '二叉树', '平衡树', '红黑树', 'B树', '字典树',
        
        # 算法类型
        '排序', '查找', '搜索', '遍历', '递归', '迭代', '分治', '动态规划',
        '贪心', '回溯', '深度优先', '广度优先', 'DFS', 'BFS',
        
        # 算法特性
        '稳定', '不稳定', '原地', '非原地', '比较', '非比较',
        '最优', '平均', '最坏', '最好',
        
        # 操作
        '插入', '删除', '查询', '更新', '合并', '分割', '交换',
        '比较', '移动', '复制',
    ]
    
    # 提取出现的概念
    concepts = []
    text_lower = text.lower()
    
    for keyword in concept_keywords:
        if keyword.lower() in text_lower or keyword in text:
            concepts.append(keyword)
    
    return list(set(concepts))  # 去重


def compute_concept_coverage(prediction: str, reference: str) -> Dict[str, float]:
    """计算概念覆盖度。
    
    评估预测答案是否覆盖了参考答案中的核心概念。
    
    Args:
        prediction: 预测答案
        reference: 参考答案
    
    Returns:
        包含覆盖度和详细信息的字典
    """
    # 提取概念
    pred_concepts = set(extract_algorithm_concepts(prediction))
    ref_concepts = set(extract_algorithm_concepts(reference))
    
    if len(ref_concepts) == 0:
        # 参考答案没有明确概念，返回中性分数
        return {
            'concept_coverage': 0.5,
            'covered_concepts': list(pred_concepts),
            'missing_concepts': [],
            'total_concepts': 0,
        }
    
    # 计算覆盖的概念
    covered = pred_concepts & ref_concepts
    missing = ref_concepts - pred_concepts
    
    coverage = len(covered) / len(ref_concepts)
    
    return {
        'concept_coverage': coverage,
        'covered_concepts': list(covered),
        'missing_concepts': list(missing),
        'total_concepts': len(ref_concepts),
    }


# ============================================================================
# 4. 解释清晰度（教育领域指标）
# ============================================================================

def compute_explanation_clarity(answer: str) -> Dict[str, float]:
    """评估答案的解释清晰度。
    
    评分标准：
    - 有明确的步骤划分：+0.25
    - 有具体的例子：+0.25
    - 有代码片段：+0.25
    - 语言通俗易懂（长度适中）：+0.25
    
    Args:
        answer: 答案文本
    
    Returns:
        包含清晰度分数和详细信息的字典
    """
    score = 0.0
    details = {}
    
    # 1. 检查步骤划分（0.25）
    step_patterns = [
        r'[1-9]\.',  # 1. 2. 3.
        r'第[一二三四五六七八九十]+步',  # 第一步、第二步
        r'首先|然后|接着|最后|其次',  # 连接词
        r'步骤[1-9]',  # 步骤1、步骤2
    ]
    
    has_steps = any(re.search(pattern, answer) for pattern in step_patterns)
    if has_steps:
        score += 0.25
        details['has_steps'] = True
    else:
        details['has_steps'] = False
    
    # 2. 检查例子（0.25）
    example_patterns = [
        r'例如|比如|举例|例子',
        r'假设|假如|设',
        r'比方说|譬如',
    ]
    
    has_examples = any(re.search(pattern, answer) for pattern in example_patterns)
    if has_examples:
        score += 0.25
        details['has_examples'] = True
    else:
        details['has_examples'] = False
    
    # 3. 检查代码片段（0.25）
    code_patterns = [
        r'```',  # Markdown 代码块
        r'`[^`]+`',  # 行内代码
        r'def |function |class ',  # 代码关键字
        r'for |while |if ',  # 控制流
    ]
    
    has_code = any(re.search(pattern, answer) for pattern in code_patterns)
    if has_code:
        score += 0.25
        details['has_code'] = True
    else:
        details['has_code'] = False
    
    # 4. 语言通俗性（长度适中）（0.25）
    # 简化版：检查长度是否在合理范围内（100-2000 字符）
    length = len(answer)
    if 100 <= length <= 2000:
        score += 0.25
        details['appropriate_length'] = True
    else:
        details['appropriate_length'] = False
    
    details['length'] = length
    
    return {
        'explanation_clarity': score,
        'details': details,
    }


# ============================================================================
# 5. 代码正确性（教育领域指标）
# ============================================================================

def compute_code_correctness(answer: str, reference: str) -> Dict[str, float]:
    """评估答案中代码的正确性。
    
    简化版：检查是否包含代码，以及代码片段的相似度。
    
    Args:
        answer: 预测答案
        reference: 参考答案
    
    Returns:
        包含代码正确性分数的字典
    """
    # 提取代码片段
    def extract_code(text: str) -> List[str]:
        # 提取 Markdown 代码块
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        # 提取行内代码
        inline_codes = re.findall(r'`[^`]+`', text)
        return code_blocks + inline_codes
    
    pred_codes = extract_code(answer)
    ref_codes = extract_code(reference)
    
    if len(ref_codes) == 0:
        # 参考答案没有代码，返回中性分数
        return {
            'code_correctness': 0.5,
            'has_code': len(pred_codes) > 0,
            'code_count': len(pred_codes),
        }
    
    if len(pred_codes) == 0:
        # 预测答案没有代码，但参考答案有
        return {
            'code_correctness': 0.0,
            'has_code': False,
            'code_count': 0,
        }
    
    # 简化版：检查代码数量是否匹配
    count_match = min(len(pred_codes), len(ref_codes)) / max(len(pred_codes), len(ref_codes))
    
    return {
        'code_correctness': count_match,
        'has_code': True,
        'code_count': len(pred_codes),
    }


# ============================================================================
# 6. 综合评分
# ============================================================================

def compute_comprehensive_score(metrics: Dict[str, float]) -> float:
    """计算综合评分。
    
    综合评分 = 自动指标 × 0.6 + 教育指标 × 0.4
    
    Args:
        metrics: 包含所有指标的字典
    
    Returns:
        综合评分（0-1）
    """
    # 自动指标（0.6）
    auto_score = 0.0
    auto_weight = 0.0
    
    if 'rouge_l' in metrics:
        auto_score += metrics['rouge_l'] * 0.4
        auto_weight += 0.4
    
    if 'bleu_4' in metrics:
        auto_score += metrics['bleu_4'] * 0.3
        auto_weight += 0.3
    
    if 'bertscore_f1' in metrics:
        auto_score += metrics['bertscore_f1'] * 0.3
        auto_weight += 0.3
    
    if auto_weight > 0:
        auto_score = auto_score / auto_weight
    
    # 教育指标（0.4）
    edu_score = 0.0
    edu_weight = 0.0
    
    if 'concept_coverage' in metrics:
        edu_score += metrics['concept_coverage'] * 0.5
        edu_weight += 0.5
    
    if 'explanation_clarity' in metrics:
        edu_score += metrics['explanation_clarity'] * 0.5
        edu_weight += 0.5
    
    if edu_weight > 0:
        edu_score = edu_score / edu_weight
    
    # 综合评分
    if auto_weight > 0 and edu_weight > 0:
        comprehensive = auto_score * 0.6 + edu_score * 0.4
    elif auto_weight > 0:
        comprehensive = auto_score
    elif edu_weight > 0:
        comprehensive = edu_score
    else:
        comprehensive = 0.0
    
    return comprehensive


# ============================================================================
# 7. 批量评测
# ============================================================================

def evaluate_answer(prediction: str, reference: str) -> Dict[str, any]:
    """对单个答案进行全面评测。
    
    Args:
        prediction: 预测答案
        reference: 参考答案
    
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    
    # 1. F1-Score
    metrics['f1_score'] = compute_f1_score(prediction, reference)
    
    # 2. 概念覆盖度
    concept_result = compute_concept_coverage(prediction, reference)
    metrics['concept_coverage'] = concept_result['concept_coverage']
    metrics['concept_details'] = {
        'covered': concept_result['covered_concepts'],
        'missing': concept_result['missing_concepts'],
        'total': concept_result['total_concepts'],
    }
    
    # 3. 解释清晰度
    clarity_result = compute_explanation_clarity(prediction)
    metrics['explanation_clarity'] = clarity_result['explanation_clarity']
    metrics['clarity_details'] = clarity_result['details']
    
    # 4. 代码正确性
    code_result = compute_code_correctness(prediction, reference)
    metrics['code_correctness'] = code_result['code_correctness']
    metrics['code_details'] = {
        'has_code': code_result['has_code'],
        'code_count': code_result['code_count'],
    }
    
    return metrics


def batch_evaluate(
    predictions: List[str],
    references: List[str],
) -> Dict[str, any]:
    """批量评测多个答案。
    
    Args:
        predictions: 预测答案列表
        references: 参考答案列表
    
    Returns:
        包含平均指标和详细结果的字典
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"预测答案数量 ({len(predictions)}) 与参考答案数量 ({len(references)}) 不匹配"
        )
    
    # 逐个评测
    results = []
    for pred, ref in zip(predictions, references):
        result = evaluate_answer(pred, ref)
        results.append(result)
    
    # 计算平均值
    avg_metrics = {
        'f1_score': sum(r['f1_score'] for r in results) / len(results),
        'concept_coverage': sum(r['concept_coverage'] for r in results) / len(results),
        'explanation_clarity': sum(r['explanation_clarity'] for r in results) / len(results),
        'code_correctness': sum(r['code_correctness'] for r in results) / len(results),
    }
    
    # 计算 BERTScore（批量计算更高效）
    bertscore_result = compute_bertscore(predictions, references)
    avg_metrics.update(bertscore_result)
    
    return {
        'average_metrics': avg_metrics,
        'detailed_results': results,
        'sample_count': len(results),
    }
