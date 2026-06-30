"""算法推荐模块

根据学习历史推荐相关算法"""

from typing import Any, Dict, List, Optional, Set
from collections import Counter, defaultdict
from .db_source import search_knowledge_nodes, fetch_relations_for_nodes


def recommend_algorithms(
    user_id: int,
    session_history: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """基于用户学习历史推荐算法
    
    参数:
        user_id: 用户ID
        session_history: 会话历史
        top_k: 推荐数量
    
    返回:
        [
            {
                'algorithm_id': 123,
                'algorithm_name': '归并排序',
                'score': 0.85,
                'reasons': ['与快速排序相关', '属于排序算法类'],
                'difficulty': 'medium'
            }
        ]
    """
    # 提取用户已学习的算法
    learned_algorithms = _extract_learned_algorithms(session_history)
    
    if not learned_algorithms:
        return _get_popular_algorithms(top_k)
    
    # 基于多种策略计算推荐分数
    candidates = {}
    
    # 策略1: 知识图谱关系（权重0.4）
    kg_recommendations = _recommend_by_knowledge_graph(learned_algorithms)
    for algo_id, score in kg_recommendations.items():
        candidates[algo_id] = candidates.get(algo_id, 0) + score * 0.4
    
    # 策略2: 难度渐进（权重0.3）
    difficulty_recommendations = _recommend_by_difficulty(learned_algorithms)
    for algo_id, score in difficulty_recommendations.items():
        candidates[algo_id] = candidates.get(algo_id, 0) + score * 0.3
    
    # 策略3: 相似主题（权重0.3）
    topic_recommendations = _recommend_by_topic(learned_algorithms)
    for algo_id, score in topic_recommendations.items():
        candidates[algo_id] = candidates.get(algo_id, 0) + score * 0.3
    
    # 过滤已学习的
    for algo_id in learned_algorithms:
        candidates.pop(algo_id, None)
    
    # 排序并返回top_k
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for algo_id, score in sorted_candidates[:top_k]:
        algo_info = _get_algorithm_info(algo_id)
        if algo_info:
            algo_info['score'] = round(score, 2)
            algo_info['reasons'] = _generate_recommendation_reasons(algo_id, learned_algorithms)
            results.append(algo_info)
    
    return results


def _extract_learned_algorithms(history: List[Dict[str, Any]]) -> List[int]:
    """从历史中提取已学习的算法ID"""
    algorithm_ids = []
    
    for entry in history:
        algo_id = entry.get('algorithm_id')
        if algo_id:
            algorithm_ids.append(int(algo_id))
    
    # 去重并保持顺序
    seen: Set[int] = set()
    unique_ids = []
    for aid in algorithm_ids:
        if aid not in seen:
            seen.add(aid)
            unique_ids.append(aid)
    
    return unique_ids


def _recommend_by_knowledge_graph(learned_algorithms: List[int]) -> Dict[int, float]:
    """基于知识图谱推荐
    
    推荐与已学算法有关系的其他算法
    """
    recommendations = {}
    
    # 查找已学算法在知识图谱中的节点
    for algo_id in learned_algorithms:
        # 查询相关节点和关系
        # （简化处理：通过算法名查询）
        relations = fetch_relations_for_nodes([algo_id], limit=20)
        
        for rel in relations:
            # 推荐目标节点
            target_id = rel['target_id'] if rel['source_id'] == algo_id else rel['source_id']
            
            # 根据关系类型设置权重
            rel_type = rel['relation_type']
            weight = 1.0
            if rel_type == '前驱算法':
                weight = 0.8  # 前驱算法权重稍低
            elif rel_type == '后继算法':
                weight = 1.2  # 后继算法权重稍高
            elif rel_type == '属于':
                weight = 0.6  # 同类算法权重中等
            elif rel_type == '优化自':
                weight = 1.1
            
            recommendations[target_id] = recommendations.get(target_id, 0) + weight
    
    # 归一化
    if recommendations:
        max_score = max(recommendations.values())
        recommendations = {k: v/max_score for k, v in recommendations.items()}
    
    return recommendations


def _recommend_by_difficulty(learned_algorithms: List[int]) -> Dict[int, float]:
    """基于难度渐进推荐
    
    推荐比已学算法稍难的算法
    """
    # 简化实现：返回空字典
    # 实际需要维护算法难度数据库
    return {}


def _recommend_by_topic(learned_algorithms: List[int]) -> Dict[int, float]:
    """基于主题相似度推荐
    
    推荐相同主题的算法
    """
    # 简化实现：返回空字典
    # 实际需要算法主题标签
    return {}


def _get_popular_algorithms(top_k: int) -> List[Dict[str, Any]]:
    """获取热门算法（新用户默认推荐）"""
    # 简化实现：返回常见算法
    popular = [
        {'algorithm_id': 1, 'algorithm_name': '快速排序', 'score': 1.0, 'reasons': ['基础算法'], 'difficulty': 'medium'},
        {'algorithm_id': 2, 'algorithm_name': '二分查找', 'score': 0.95, 'reasons': ['基础算法'], 'difficulty': 'easy'},
        {'algorithm_id': 3, 'algorithm_name': '动态规划', 'score': 0.9, 'reasons': ['重要算法'], 'difficulty': 'hard'},
        {'algorithm_id': 4, 'algorithm_name': 'BFS', 'score': 0.85, 'reasons': ['图算法基础'], 'difficulty': 'medium'},
        {'algorithm_id': 5, 'algorithm_name': 'DFS', 'score': 0.85, 'reasons': ['图算法基础'], 'difficulty': 'medium'},
    ]
    
    return popular[:top_k]


def _get_algorithm_info(algo_id: int) -> Optional[Dict[str, Any]]:
    """获取算法基本信息"""
    # 简化实现：返回模拟数据
    # 实际应该查询数据库
    return {
        'algorithm_id': algo_id,
        'algorithm_name': f'算法{algo_id}',
        'difficulty': 'medium'
    }


def _generate_recommendation_reasons(algo_id: int, learned_algorithms: List[int]) -> List[str]:
    """生成推荐理由"""
    reasons = []
    
    # 查询知识图谱关系
    for learned_id in learned_algorithms:
        relations = fetch_relations_for_nodes([learned_id, algo_id], limit=10)
        for rel in relations:
            if (rel['source_id'] == learned_id and rel['target_id'] == algo_id) or \
               (rel['target_id'] == learned_id and rel['source_id'] == algo_id):
                reasons.append(f"与{rel['source_name']}有{rel['relation_type']}关系")
                break
    
    if not reasons:
        reasons.append('系统推荐')
    
    return reasons[:3]  # 最多3个理由


def get_learning_path(
    start_algorithm: str,
    end_algorithm: str
) -> Dict[str, Any]:
    """生成从start到end的学习路径
    
    返回:
        {
            'path': ['算法1', '算法2', '算法3'],
            'steps': [
                {'algorithm': '算法1', 'reason': '...', 'estimated_time': '2小时'},
                ...
            ],
            'total_time': '6小时'
        }
    """
    # 简化实现：BFS查找最短路径
    # 实际应该考虑难度递增、知识依赖等
    
    # 查找起点和终点节点
    start_nodes = search_knowledge_nodes(start_algorithm, limit=1)
    end_nodes = search_knowledge_nodes(end_algorithm, limit=1)
    
    if not start_nodes or not end_nodes:
        return {'path': [], 'message': '算法未找到'}
    
    start_id = start_nodes[0]['id']
    end_id = end_nodes[0]['id']
    
    # BFS查找路径
    queue = [(start_id, [start_id])]
    visited = {start_id}
    
    for _ in range(5):  # 最多5跳
        if not queue:
            break
        
        current_id, path = queue.pop(0)
        
        if current_id == end_id:
            # 找到路径
            return _build_learning_path(path)
        
        # 扩展邻居
        relations = fetch_relations_for_nodes([current_id], limit=20)
        
        for rel in relations:
            # 优先考虑"后继算法"和"优化自"关系
            if rel['relation_type'] in ['后继算法', '优化自']:
                neighbor_id = rel['target_id'] if rel['source_id'] == current_id else rel['source_id']
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
    
    return {'path': [], 'message': '未找到合适的学习路径'}


def _build_learning_path(path_ids: List[int]) -> Dict[str, Any]:
    """构建完整的学习路径"""
    steps = []
    
    for i, algo_id in enumerate(path_ids, 1):
        algo_info = _get_algorithm_info(algo_id)
        if algo_info:
            steps.append({
                'step': i,
                'algorithm': algo_info['algorithm_name'],
                'algorithm_id': algo_id,
                'reason': _get_step_reason(i, len(path_ids)),
                'estimated_time': '2-3小时',
                'difficulty': algo_info.get('difficulty', 'medium')
            })
    
    return {
        'path': [s['algorithm'] for s in steps],
        'steps': steps,
        'total_steps': len(steps),
        'total_time': f'{len(steps) * 2}-{len(steps) * 3}小时'
    }


def _get_step_reason(step_num: int, total_steps: int) -> str:
    """生成学习步骤的理由"""
    if step_num == 1:
        return '起点算法，建立基础'
    elif step_num == total_steps:
        return '目标算法'
    elif step_num == 2:
        return '建立核心概念'
    else:
        return f'第{step_num}步：逐步深入'
