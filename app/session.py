"""会话管理模块

负责管理用户会话历史、上下文追踪和会话存储。
"""

from datetime import datetime
from typing import Dict, List, Optional
import json


# 简单的内存存储（生产环境应该用Redis）
_sessions: Dict[str, List[Dict]] = {}


def save_session_history(
    session_id: str,
    user_id: Optional[int],
    algorithm_id: int,
    question: str,
    answer: str,
    references: Optional[List[Dict]] = None,
) -> None:
    """保存会话历史
    
    参数:
        session_id: 会话ID
        user_id: 用户ID（可选）
        algorithm_id: 算法ID
        question: 用户问题
        answer: 系统回答
        references: 引用的chunk列表
    """
    if session_id not in _sessions:
        _sessions[session_id] = []
    
    _sessions[session_id].append({
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'algorithm_id': algorithm_id,
        'question': question,
        'answer': answer,
        'references': references or []
    })
    
    # 限制历史记录数量（最多保留20条）
    if len(_sessions[session_id]) > 20:
        _sessions[session_id] = _sessions[session_id][-20:]


def get_session_history(session_id: str, limit: int = 10) -> List[Dict]:
    """获取会话历史
    
    参数:
        session_id: 会话ID
        limit: 返回最近N条记录
    
    返回:
        会话历史列表
    """
    history = _sessions.get(session_id, [])
    return history[-limit:] if limit > 0 else history


def get_last_algorithm(session_id: str) -> Optional[int]:
    """获取会话中最后讨论的算法ID"""
    history = get_session_history(session_id, limit=1)
    if history:
        return history[-1].get('algorithm_id')
    return None


def clear_session(session_id: str) -> None:
    """清空会话历史"""
    if session_id in _sessions:
        del _sessions[session_id]


def get_session_algorithms(session_id: str) -> List[int]:
    """获取会话中讨论过的所有算法ID（去重）"""
    history = get_session_history(session_id, limit=0)  # 获取全部
    algorithm_ids = [h['algorithm_id'] for h in history if h.get('algorithm_id')]
    return list(dict.fromkeys(algorithm_ids))  # 去重且保持顺序


def get_session_summary(session_id: str) -> Dict:
    """获取会话摘要信息"""
    history = get_session_history(session_id, limit=0)
    
    if not history:
        return {
            'session_id': session_id,
            'total_messages': 0,
            'algorithms_discussed': [],
            'created_at': None,
            'last_activity': None
        }
    
    return {
        'session_id': session_id,
        'total_messages': len(history),
        'algorithms_discussed': get_session_algorithms(session_id),
        'created_at': history[0]['timestamp'] if history else None,
        'last_activity': history[-1]['timestamp'] if history else None,
        'user_id': history[-1].get('user_id')
    }
