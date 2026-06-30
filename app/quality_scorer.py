"""RAG质量评分模块

实时评估答案质量"""

from typing import Any, Dict, List
import re


def score_answer_quality(
    question: str,
    answer: str,
    references: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """综合评估答案质量
    
    返回:
        {
            'total_score': 总分 (0-100),
            'scores': {
                'completeness': 完整性得分,
                'relevance': 相关性得分,
                'clarity': 清晰度得分,
                'citation': 引用质量得分
            },
            'feedback': {
                'strengths': ['优点1', '优点2'],
                'improvements': ['改进建议1', '改进建议2']
            },
            'grade': 'A/B/C/D/F'
        }
    """
    scores = {}
    
    # 1. 完整性评分 (30分)
    scores['completeness'] = _score_completeness(question, answer)
    
    # 2. 相关性评分 (25分)
    scores['relevance'] = _score_relevance(question, answer)
    
    # 3. 清晰度评分 (25分)
    scores['clarity'] = _score_clarity(answer)
    
    # 4. 引用质量评分 (20分)
    scores['citation'] = _score_citation(answer, references)
    
    # 计算总分
    weights = {
        'completeness': 0.30,
        'relevance': 0.25,
        'clarity': 0.25,
        'citation': 0.20
    }
    
    total_score = sum(scores[k] * weights[k] for k in scores)
    
    # 生成反馈
    feedback = _generate_feedback(scores, answer, references)
    
    # 评级
    grade = _calculate_grade(total_score)
    
    return {
        'total_score': round(total_score, 1),
        'scores': {k: round(v, 1) for k, v in scores.items()},
        'feedback': feedback,
        'grade': grade
    }


def _score_completeness(question: str, answer: str) -> float:
    """评估答案完整性 (0-100)
    
    考虑因素:
    - 答案长度是否适中
    - 是否包含必要的要素（原理、步骤、示例等）
    - 是否回答了问题的所有方面
    """
    score = 50.0  # 基础分
    
    # 长度评分
    answer_len = len(answer)
    if answer_len < 50:
        score -= 30  # 太短
    elif answer_len < 100:
        score -= 15
    elif 200 <= answer_len <= 800:
        score += 20  # 长度适中
    elif answer_len > 1500:
        score -= 10  # 可能过长
    
    # 结构化评分
    has_structure = any(marker in answer for marker in ['1.', '2.', '首先', '其次', '最后'])
    if has_structure:
        score += 15
    
    # 关键词覆盖
    q_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', question))
    a_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', answer))
    coverage = len(q_keywords & a_keywords) / max(len(q_keywords), 1)
    score += coverage * 15
    
    return min(100, max(0, score))


def _score_relevance(question: str, answer: str) -> float:
    """评估答案相关性 (0-100)
    
    考虑因素:
    - 是否直接回答问题
    - 是否偏题
    """
    score = 60.0  # 基础分
    
    # 关键词匹配
    q_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', question))
    a_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', answer))
    
    if not q_keywords:
        return score
    
    match_ratio = len(q_keywords & a_keywords) / len(q_keywords)
    score += match_ratio * 30
    
    # 检查是否包含无关内容
    unrelated_markers = ['顺便说一下', '另外', '题外话']
    if any(marker in answer for marker in unrelated_markers):
        score -= 10
    
    return min(100, max(0, score))


def _score_clarity(answer: str) -> float:
    """评估答案清晰度 (0-100)
    
    考虑因素:
    - 结构是否清晰
    - 是否有分段
    - 是否有重复
    """
    score = 50.0  # 基础分
    
    # 分段评分
    paragraphs = [p for p in answer.split('\n') if p.strip()]
    if len(paragraphs) >= 3:
        score += 20
    elif len(paragraphs) >= 2:
        score += 10
    
    # 序号/标记评分
    has_numbering = bool(re.search(r'\d+\.', answer))
    has_keywords = any(kw in answer for kw in ['首先', '其次', '然后', '最后'])
    if has_numbering or has_keywords:
        score += 15
    
    # 重复检查（简单检测）
    sentences = answer.split('。')
    if len(sentences) > len(set(sentences)) * 1.2:
        score -= 15  # 有重复
    
    # 专业术语使用
    technical_terms = len(re.findall(r'[A-Z]{2,}|O\([^)]+\)', answer))
    score += min(15, technical_terms * 3)
    
    return min(100, max(0, score))


def _score_citation(answer: str, references: List[Dict[str, Any]]) -> float:
    """评估引用质量 (0-100)
    
    考虑因素:
    - 是否有引用标记
    - 引用是否准确
    - 引用数量是否适当
    """
    score = 50.0  # 基础分
    
    # 检测引用标记
    citations = re.findall(r'\[chunk_id=(\d+)\]', answer)
    
    if not citations:
        return 30  # 无引用，基础分较低
    
    # 引用数量评分
    citation_count = len(set(citations))
    if 1 <= citation_count <= 5:
        score += 30
    elif citation_count > 5:
        score += 20  # 引用过多
    
    # 引用准确性（检查引用的chunk_id是否在references中）
    ref_ids = {r.get('id') for r in references}
    valid_citations = sum(1 for cid in citations if int(cid) in ref_ids)
    if citations:
        accuracy = valid_citations / len(citations)
        score += accuracy * 20
    
    return min(100, max(0, score))


def _generate_feedback(
    scores: Dict[str, float],
    answer: str,
    references: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """生成改进建议"""
    strengths = []
    improvements = []
    
    # 完整性反馈
    if scores['completeness'] >= 80:
        strengths.append('答案内容完整，覆盖了问题的主要方面')
    elif scores['completeness'] < 60:
        improvements.append('答案内容不够完整，建议补充更多细节')
    
    # 相关性反馈
    if scores['relevance'] >= 80:
        strengths.append('答案紧扣问题主题，相关性强')
    elif scores['relevance'] < 60:
        improvements.append('答案与问题关联不够紧密，建议聚焦核心内容')
    
    # 清晰度反馈
    if scores['clarity'] >= 80:
        strengths.append('答案结构清晰，易于理解')
    elif scores['clarity'] < 60:
        improvements.append('答案结构不够清晰，建议增加分段和序号')
    
    # 引用反馈
    if scores['citation'] >= 80:
        strengths.append('引用标注规范，便于溯源')
    elif scores['citation'] < 60:
        improvements.append('建议在关键句后添加引用标记[chunk_id=xx]')
    
    return {
        'strengths': strengths,
        'improvements': improvements
    }


def _calculate_grade(total_score: float) -> str:
    """计算等级"""
    if total_score >= 90:
        return 'A'
    elif total_score >= 80:
        return 'B'
    elif total_score >= 70:
        return 'C'
    elif total_score >= 60:
        return 'D'
    else:
        return 'F'


def compare_answer_versions(
    answers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """对比多个答案版本的质量
    
    参数:
        answers: [
            {'version': 'v1', 'answer': '...', 'references': [...]},
            {'version': 'v2', 'answer': '...', 'references': [...]}
        ]
    
    返回:
        {
            'best_version': 'v1',
            'comparison': [
                {'version': 'v1', 'score': 85, 'rank': 1},
                ...
            ],
            'recommendation': '建议使用v1版本...'
        }
    """
    results = []
    
    for item in answers:
        question = item.get('question', '')
        score_result = score_answer_quality(
            question,
            item['answer'],
            item.get('references', [])
        )
        
        results.append({
            'version': item.get('version', 'unknown'),
            'score': score_result['total_score'],
            'grade': score_result['grade'],
            'scores': score_result['scores']
        })
    
    # 排序
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results, 1):
        r['rank'] = i
    
    best = results[0]
    
    recommendation = f"建议使用{best['version']}版本（评分{best['score']}，等级{best['grade']}）"
    if len(results) > 1:
        score_diff = best['score'] - results[1]['score']
        if score_diff < 5:
            recommendation += f"，但与{results[1]['version']}版本差距较小（{score_diff:.1f}分）"
    
    return {
        'best_version': best['version'],
        'comparison': results,
        'recommendation': recommendation
    }
