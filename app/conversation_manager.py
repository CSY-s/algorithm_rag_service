"""对话分支管理模块

支持对话回退、分支创建、历史浏览"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json


# 内存存储（生产环境建议用Redis）
_conversations: Dict[str, Dict[str, Any]] = {}


def create_branch(
    session_id: str,
    branch_point: int,
    new_question: str
) -> Dict[str, Any]:
    """从某个历史点创建新分支
    
    参数:
        session_id: 会话ID
        branch_point: 分支点的消息索引
        new_question: 新分支的第一个问题
    
    返回:
        {
            'branch_id': '新分支ID',
            'parent_session': 'session_id',
            'branch_from': branch_point,
            'message': '成功创建分支'
        }
    """
    if session_id not in _conversations:
        return {'error': '会话不存在'}
    
    conv = _conversations[session_id]
    history = conv.get('history', [])
    
    if branch_point < 0 or branch_point >= len(history):
        return {'error': '分支点无效'}
    
    # 创建新分支ID
    branch_id = f"{session_id}_branch_{len(conv.get('branches', []))+ 1}"
    
    # 复制历史到分支点
    branch_history = history[:branch_point + 1]
    
    # 创建新分支
    _conversations[branch_id] = {
        'session_id': branch_id,
        'parent_session': session_id,
        'branch_from': branch_point,
        'history': branch_history,
        'created_at': datetime.now().isoformat(),
        'branches': []
    }
    
    # 在父会话中记录分支
    if 'branches' not in conv:
        conv['branches'] = []
    conv['branches'].append({
        'branch_id': branch_id,
        'branch_from': branch_point,
        'first_question': new_question,
        'created_at': datetime.now().isoformat()
    })
    
    return {
        'branch_id': branch_id,
        'parent_session': session_id,
        'branch_from': branch_point,
        'history_size': len(branch_history),
        'message': '成功创建分支'
    }


def rollback_to_message(
    session_id: str,
    message_index: int
) -> Dict[str, Any]:
    """回退到某个历史消息
    
    参数:
        session_id: 会话ID
        message_index: 要回退到的消息索引
    
    返回:
        {
            'session_id': 'session_id',
            'rollback_to': message_index,
            'removed_count': 删除的消息数,
            'current_history': [...]
        }
    """
    if session_id not in _conversations:
        return {'error': '会话不存在'}
    
    conv = _conversations[session_id]
    history = conv.get('history', [])
    
    if message_index < 0 or message_index >= len(history):
        return {'error': '索引无效'}
    
    # 保存被删除的历史（用于可能的恢复）
    removed = history[message_index + 1:]
    if 'rollback_history' not in conv:
        conv['rollback_history'] = []
    conv['rollback_history'].append({
        'rollback_at': datetime.now().isoformat(),
        'rollback_to': message_index,
        'removed_messages': removed
    })
    
    # 截断历史
    conv['history'] = history[:message_index + 1]
    
    return {
        'session_id': session_id,
        'rollback_to': message_index,
        'removed_count': len(removed),
        'current_history': conv['history'],
        'message': f'已回退到消息{message_index}，删除了{len(removed)}条后续消息'
    }


def get_conversation_tree(session_id: str) -> Dict[str, Any]:
    """获取完整的对话树（包含所有分支）
    
    返回:
        {
            'root_session': 'session_id',
            'main_history': [...],
            'branches': [
                {
                    'branch_id': '...',
                    'branch_from': 消息索引,
                    'history': [...],
                    'sub_branches': [...]
                }
            ]
        }
    """
    if session_id not in _conversations:
        return {'error': '会话不存在'}
    
    conv = _conversations[session_id]
    
    def build_branch_tree(sid: str) -> Dict[str, Any]:
        if sid not in _conversations:
            return {}
        
        c = _conversations[sid]
        branches_info = []
        
        for branch in c.get('branches', []):
            branch_id = branch['branch_id']
            branch_tree = build_branch_tree(branch_id)
            branches_info.append({
                **branch,
                'history_size': len(_conversations.get(branch_id, {}).get('history', [])),
                'sub_branches': branch_tree.get('branches', [])
            })
        
        return {
            'session_id': sid,
            'history_size': len(c.get('history', [])),
            'branches': branches_info
        }
    
    tree = build_branch_tree(session_id)
    tree['main_history'] = conv.get('history', [])
    
    return tree


def compare_branches(
    session_id: str,
    branch_id1: str,
    branch_id2: str
) -> Dict[str, Any]:
    """对比两个分支的差异
    
    返回:
        {
            'common_prefix': 共同前缀长度,
            'branch1_unique': 分支1独有的消息数,
            'branch2_unique': 分支2独有的消息数,
            'divergence_point': 分叉点索引
        }
    """
    if branch_id1 not in _conversations or branch_id2 not in _conversations:
        return {'error': '分支不存在'}
    
    history1 = _conversations[branch_id1].get('history', [])
    history2 = _conversations[branch_id2].get('history', [])
    
    # 找到共同前缀
    common_length = 0
    for i in range(min(len(history1), len(history2))):
        if history1[i] == history2[i]:
            common_length += 1
        else:
            break
    
    return {
        'common_prefix': common_length,
        'branch1_unique': len(history1) - common_length,
        'branch2_unique': len(history2) - common_length,
        'divergence_point': common_length - 1 if common_length > 0 else None
    }


def undo_last_rollback(session_id: str) -> Dict[str, Any]:
    """撤销最后一次回退操作
    
    返回:
        {
            'restored_count': 恢复的消息数,
            'current_history': [...]
        }
    """
    if session_id not in _conversations:
        return {'error': '会话不存在'}
    
    conv = _conversations[session_id]
    rollback_history = conv.get('rollback_history', [])
    
    if not rollback_history:
        return {'error': '没有可撤销的回退操作'}
    
    # 取最后一次回退
    last_rollback = rollback_history.pop()
    removed_messages = last_rollback['removed_messages']
    
    # 恢复消息
    conv['history'].extend(removed_messages)
    
    return {
        'restored_count': len(removed_messages),
        'current_history': conv['history'],
        'message': f'已恢复{len(removed_messages)}条消息'
    }


# 内部辅助函数：会话初始化
def _init_session(session_id: str):
    """初始化会话（如果不存在）"""
    if session_id not in _conversations:
        _conversations[session_id] = {
            'session_id': session_id,
            'history': [],
            'branches': [],
            'created_at': datetime.now().isoformat()
        }


# 内部辅助函数：添加消息
def _add_message(session_id: str, message: Dict[str, Any]):
    """添加消息到会话"""
    _init_session(session_id)
    _conversations[session_id]['history'].append(message)
