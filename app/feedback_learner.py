"""用户反馈学习模块

根据用户点赞/点踩调整系统"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

# 内存存储（生产环境建议用数据库）
_feedback_store: Dict[str, List[Dict[str, Any]]] = {}
_feedback_stats: Dict[str, Dict[str, int]] = {}


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    feedback_type: str,  # 'thumbs_up' or 'thumbs_down'
    feedback_reason: Optional[str] = None,
    chunk_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """记录用户反馈
    
    参数:
        session_id: 会话ID
        question: 问题
        answer: 答案
        feedback_type: 反馈类型 (thumbs_up/thumbs_down)
        feedback_reason: 反馈原因（可选）
        chunk_ids: 相关chunk ID列表
    
    返回:
        {'feedback_id': '...', 'recorded': True}
    """
    feedback_id = f"{session_id}_{datetime.now().timestamp()}"
    
    feedback_record = {
        'feedback_id': feedback_id,
        'session_id': session_id,
        'question': question,
        'answer': answer,
        'feedback_type': feedback_type,
        'feedback_reason': feedback_reason,
        'chunk_ids': chunk_ids or [],
        'timestamp': datetime.now().isoformat()
    }
    
    # 存储反馈
    if session_id not in _feedback_store:
        _feedback_store[session_id] = []
    _feedback_store[session_id].append(feedback_record)
    
    # 更新统计
    _update_feedback_stats(feedback_type, chunk_ids)
    
    return {
        'feedback_id': feedback_id,
        'recorded': True,
        'message': '感谢您的反馈！'
    }


def _update_feedback_stats(feedback_type: str, chunk_ids: Optional[List[int]]):
    """更新反馈统计"""
    # 全局统计
    if 'global' not in _feedback_stats:
        _feedback_stats['global'] = {'thumbs_up': 0, 'thumbs_down': 0}
    _feedback_stats['global'][feedback_type] = _feedback_stats['global'].get(feedback_type, 0) + 1
    
    # chunk级别统计
    if chunk_ids:
        for cid in chunk_ids:
            cid_str = str(cid)
            if cid_str not in _feedback_stats:
                _feedback_stats[cid_str] = {'thumbs_up': 0, 'thumbs_down': 0}
            _feedback_stats[cid_str][feedback_type] = _feedback_stats[cid_str].get(feedback_type, 0) + 1


def get_feedback_stats(chunk_id: Optional[int] = None) -> Dict[str, Any]:
    """获取反馈统计
    
    参数:
        chunk_id: 可选，获取特定chunk的统计
    
    返回:
        统计信息
    """
    if chunk_id:
        cid_str = str(chunk_id)
        stats = _feedback_stats.get(cid_str, {'thumbs_up': 0, 'thumbs_down': 0})
        total = stats['thumbs_up'] + stats['thumbs_down']
        satisfaction = stats['thumbs_up'] / total if total > 0 else 0
        
        return {
            'chunk_id': chunk_id,
            'thumbs_up': stats['thumbs_up'],
            'thumbs_down': stats['thumbs_down'],
            'total_feedback': total,
            'satisfaction_rate': round(satisfaction, 2)
        }
    else:
        # 全局统计
        stats = _feedback_stats.get('global', {'thumbs_up': 0, 'thumbs_down': 0})
        total = stats['thumbs_up'] + stats['thumbs_down']
        satisfaction = stats['thumbs_up'] / total if total > 0 else 0
        
        return {
            'global_thumbs_up': stats['thumbs_up'],
            'global_thumbs_down': stats['thumbs_down'],
            'total_feedback': total,
            'satisfaction_rate': round(satisfaction, 2)
        }


def get_low_quality_chunks(threshold: float = 0.5, min_feedback: int = 3) -> List[Dict[str, Any]]:
    """获取低质量chunk列表（满意度低于阈值）
    
    参数:
        threshold: 满意度阈值（0-1）
        min_feedback: 最少反馈数（避免样本太少）
    
    返回:
        [{
            'chunk_id': 123,
            'satisfaction_rate': 0.3,
            'thumbs_up': 3,
            'thumbs_down': 7,
            'priority': 'high'  # high/medium/low
        }]
    """
    low_quality = []
    
    for cid_str, stats in _feedback_stats.items():
        if cid_str == 'global':
            continue
        
        total = stats['thumbs_up'] + stats['thumbs_down']
        if total < min_feedback:
            continue
        
        satisfaction = stats['thumbs_up'] / total
        if satisfaction < threshold:
            priority = 'high' if satisfaction < 0.3 else ('medium' if satisfaction < 0.5 else 'low')
            
            low_quality.append({
                'chunk_id': int(cid_str),
                'satisfaction_rate': round(satisfaction, 2),
                'thumbs_up': stats['thumbs_up'],
                'thumbs_down': stats['thumbs_down'],
                'total_feedback': total,
                'priority': priority
            })
    
    # 按满意度排序
    low_quality.sort(key=lambda x: x['satisfaction_rate'])
    
    return low_quality


def adjust_retrieval_weights(feedback_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """根据反馈调整检索权重
    
    参数:
        feedback_data: 反馈数据列表
    
    返回:
        调整后的权重建议
    """
    # 分析反馈模式
    positive_chunks = []
    negative_chunks = []
    
    for fb in feedback_data:
        if fb['feedback_type'] == 'thumbs_up':
            positive_chunks.extend(fb.get('chunk_ids', []))
        else:
            negative_chunks.extend(fb.get('chunk_ids', []))
    
    # 计算chunk类型分布（简化版）
    # 实际应该查询数据库获取chunk_type
    
    # 默认权重
    weights = {
        'tfidf': 0.35,
        'keyword': 0.25,
        'vector': 0.40
    }
    
    # 如果负面反馈多，建议增加向量检索权重（语义理解）
    if len(negative_chunks) > len(positive_chunks):
        weights['vector'] += 0.05
        weights['tfidf'] -= 0.03
        weights['keyword'] -= 0.02
    
    return {
        'suggested_weights': weights,
        'reason': f'基于{len(feedback_data)}条反馈分析，建议调整检索权重',
        'positive_feedback': len([f for f in feedback_data if f['feedback_type'] == 'thumbs_up']),
        'negative_feedback': len([f for f in feedback_data if f['feedback_type'] == 'thumbs_down'])
    }


def generate_improvement_report() -> Dict[str, Any]:
    """生成改进建议报告
    
    返回:
        {
            'overall_satisfaction': 0.75,
            'total_feedback': 100,
            'low_quality_chunks': [...],
            'common_issues': [...],
            'recommendations': [...]
        }
    """
    # 全局统计
    global_stats = get_feedback_stats()
    
    # 低质量chunk
    low_quality = get_low_quality_chunks()
    
    # 常见问题（从反馈原因中提取）
    common_issues = _extract_common_issues()
    
    # 生成建议
    recommendations = []
    
    if global_stats['satisfaction_rate'] < 0.7:
        recommendations.append('整体满意度较低，建议优化答案生成质量')
    
    if len(low_quality) > 10:
        recommendations.append(f'有{len(low_quality)}个chunk质量较低，建议重新标注或删除')
    
    if common_issues.get('incomplete', 0) > 5:
        recommendations.append('多个反馈提到"不完整"，建议增加答案详细度')
    
    if common_issues.get('irrelevant', 0) > 5:
        recommendations.append('多个反馈提到"不相关"，建议优化检索算法')
    
    return {
        'overall_satisfaction': global_stats['satisfaction_rate'],
        'total_feedback': global_stats['total_feedback'],
        'low_quality_chunks': low_quality[:10],  # 返回前10个
        'common_issues': common_issues,
        'recommendations': recommendations,
        'generated_at': datetime.now().isoformat()
    }


def _extract_common_issues() -> Dict[str, int]:
    """从反馈原因中提取常见问题"""
    issues = {
        'incomplete': 0,
        'irrelevant': 0,
        'incorrect': 0,
        'unclear': 0,
        'too_long': 0,
        'too_short': 0
    }
    
    for session_feedbacks in _feedback_store.values():
        for fb in session_feedbacks:
            if fb['feedback_type'] != 'thumbs_down':
                continue
            
            reason = (fb.get('feedback_reason') or '').lower()
            
            if any(kw in reason for kw in ['不完整', '缺少', '不够详细']):
                issues['incomplete'] += 1
            if any(kw in reason for kw in ['不相关', '偏题', '答非所问']):
                issues['irrelevant'] += 1
            if any(kw in reason for kw in ['错误', '不对', '不准确']):
                issues['incorrect'] += 1
            if any(kw in reason for kw in ['不清楚', '难懂', '混乱']):
                issues['unclear'] += 1
            if any(kw in reason for kw in ['太长', '啰嗦']):
                issues['too_long'] += 1
            if any(kw in reason for kw in ['太短', '太简单']):
                issues['too_short'] += 1
    
    return issues
