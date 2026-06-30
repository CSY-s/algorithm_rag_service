"""知识图谱可视化模块

提供图谱数据的可视化格式转换"""

from typing import Any, Dict, List, Optional, Set
from .db_source import (
    search_knowledge_nodes,
    fetch_relations_for_nodes,
    fetch_all_knowledge_nodes,
    fetch_all_knowledge_relations
)


def get_graph_for_query(query: str, max_nodes: int = 20) -> Dict[str, Any]:
    """根据查询获取相关的知识图谱子图
    
    参数:
        query: 查询关键词
        max_nodes: 最多返回节点数
    
    返回:
        {
            'nodes': [...],
            'edges': [...],
            'stats': {...}
        }
    """
    # 搜索相关节点
    nodes = search_knowledge_nodes(query, limit=max_nodes)
    
    if not nodes:
        return {'nodes': [], 'edges': [], 'stats': {'node_count': 0, 'edge_count': 0}}
    
    # 获取节点间的关系
    node_ids = [n['id'] for n in nodes]
    relations = fetch_relations_for_nodes(node_ids, limit=max_nodes * 2)
    
    # 转换为可视化格式
    vis_nodes = []
    for node in nodes:
        vis_nodes.append({
            'id': str(node['id']),
            'label': node['name'],
            'type': node.get('type', '未知'),
            'description': node.get('description', ''),
            'group': _classify_node_group(node.get('type', ''))
        })
    
    vis_edges = []
    for rel in relations:
        vis_edges.append({
            'from': str(rel['source_id']),
            'to': str(rel['target_id']),
            'label': rel['relation_type'],
            'type': rel['relation_type']
        })
    
    return {
        'nodes': vis_nodes,
        'edges': vis_edges,
        'stats': {
            'node_count': len(vis_nodes),
            'edge_count': len(vis_edges),
            'query': query
        }
    }


def get_algorithm_subgraph(algorithm_name: str, depth: int = 2) -> Dict[str, Any]:
    """获取特定算法的相关子图
    
    参数:
        algorithm_name: 算法名称
        depth: 扩展深度（几跳邻居）
    
    返回:
        包含该算法及其相关节点的子图
    """
    # 查找算法节点
    nodes = search_knowledge_nodes(algorithm_name, limit=5)
    
    if not nodes:
        return {'nodes': [], 'edges': [], 'message': f'未找到算法: {algorithm_name}'}
    
    # 取最匹配的节点作为中心
    center_node = nodes[0]
    
    # BFS扩展邻居
    visited_node_ids: Set[int] = {center_node['id']}
    all_nodes = [center_node]
    all_relations = []
    
    current_layer = [center_node['id']]
    
    for _ in range(depth):
        if not current_layer:
            break
        
        # 获取当前层的所有关系
        relations = fetch_relations_for_nodes(current_layer, limit=100)
        
        # 找出新的邻居节点
        next_layer = []
        for rel in relations:
            all_relations.append(rel)
            
            # 添加新的源节点
            if rel['source_id'] not in visited_node_ids:
                visited_node_ids.add(rel['source_id'])
                next_layer.append(rel['source_id'])
            
            # 添加新的目标节点
            if rel['target_id'] not in visited_node_ids:
                visited_node_ids.add(rel['target_id'])
                next_layer.append(rel['target_id'])
        
        current_layer = next_layer
    
    # 获取所有节点的详细信息
    # （简化处理：复用已有的search_knowledge_nodes）
    for node_id in visited_node_ids:
        if node_id != center_node['id']:
            # 这里简化处理，实际应该批量查询
            pass
    
    # 转换为可视化格式
    vis_nodes = []
    node_id_set = {rel['source_id'] for rel in all_relations} | {rel['target_id'] for rel in all_relations}
    node_id_set.add(center_node['id'])
    
    # 构建节点映射
    node_map = {center_node['id']: center_node}
    
    for node_id in node_id_set:
        if node_id not in node_map:
            # 从关系中提取节点信息
            for rel in all_relations:
                if rel['source_id'] == node_id:
                    node_map[node_id] = {
                        'id': node_id,
                        'name': rel['source_name'],
                        'type': '未知'
                    }
                    break
                elif rel['target_id'] == node_id:
                    node_map[node_id] = {
                        'id': node_id,
                        'name': rel['target_name'],
                        'type': '未知'
                    }
                    break
    
    for node_id, node in node_map.items():
        vis_nodes.append({
            'id': str(node_id),
            'label': node['name'],
            'type': node.get('type', '未知'),
            'description': node.get('description', ''),
            'group': _classify_node_group(node.get('type', '')),
            'is_center': node_id == center_node['id']
        })
    
    vis_edges = []
    for rel in all_relations:
        vis_edges.append({
            'from': str(rel['source_id']),
            'to': str(rel['target_id']),
            'label': rel['relation_type'],
            'type': rel['relation_type']
        })
    
    return {
        'nodes': vis_nodes,
        'edges': vis_edges,
        'center': str(center_node['id']),
        'stats': {
            'node_count': len(vis_nodes),
            'edge_count': len(vis_edges),
            'depth': depth,
            'algorithm': algorithm_name
        }
    }


def get_full_graph(limit_nodes: Optional[int] = None) -> Dict[str, Any]:
    """获取完整知识图谱
    
    警告：可能返回大量数据，建议设置limit
    
    参数:
        limit_nodes: 限制节点数量
    
    返回:
        完整图谱数据
    """
    nodes = fetch_all_knowledge_nodes(limit=limit_nodes)
    
    if not nodes:
        return {'nodes': [], 'edges': [], 'stats': {'node_count': 0, 'edge_count': 0}}
    
    node_ids = [n['id'] for n in nodes]
    relations = fetch_all_knowledge_relations(node_ids=node_ids)
    
    # 转换格式
    vis_nodes = []
    for node in nodes:
        vis_nodes.append({
            'id': str(node['id']),
            'label': node['name'],
            'type': node.get('type', '未知'),
            'description': node.get('description', ''),
            'group': _classify_node_group(node.get('type', ''))
        })
    
    vis_edges = []
    for rel in relations:
        vis_edges.append({
            'from': str(rel['source_id']),
            'to': str(rel['target_id']),
            'label': rel['relation_type'],
            'type': rel['relation_type']
        })
    
    # 统计信息
    node_types = {}
    for node in nodes:
        node_type = node.get('type', '未知')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    relation_types = {}
    for rel in relations:
        rel_type = rel['relation_type']
        relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
    
    return {
        'nodes': vis_nodes,
        'edges': vis_edges,
        'stats': {
            'node_count': len(vis_nodes),
            'edge_count': len(vis_edges),
            'node_types': node_types,
            'relation_types': relation_types
        }
    }


def _classify_node_group(node_type: str) -> int:
    """将节点类型映射到分组编号（用于可视化着色）"""
    type_map = {
        '算法': 1,
        '数据结构': 2,
        '概念': 3,
        '应用': 4,
        '问题': 5
    }
    return type_map.get(node_type, 0)


def get_shortest_path(
    from_node_name: str,
    to_node_name: str,
    max_depth: int = 5
) -> Dict[str, Any]:
    """查找两个节点间的最短路径
    
    参数:
        from_node_name: 起始节点名称
        to_node_name: 目标节点名称
        max_depth: 最大搜索深度
    
    返回:
        {
            'path': [node_id1, node_id2, ...],
            'path_nodes': [...],
            'path_edges': [...],
            'length': 路径长度
        }
    """
    # 查找起始和目标节点
    from_nodes = search_knowledge_nodes(from_node_name, limit=1)
    to_nodes = search_knowledge_nodes(to_node_name, limit=1)
    
    if not from_nodes or not to_nodes:
        return {'path': [], 'message': '节点未找到'}
    
    from_id = from_nodes[0]['id']
    to_id = to_nodes[0]['id']
    
    # BFS查找最短路径
    queue = [(from_id, [from_id])]
    visited = {from_id}
    
    for _ in range(max_depth):
        if not queue:
            break
        
        current_id, path = queue.pop(0)
        
        if current_id == to_id:
            # 找到目标，构建路径
            return _build_path_result(path)
        
        # 扩展邻居
        relations = fetch_relations_for_nodes([current_id], limit=50)
        
        for rel in relations:
            # 检查出边
            if rel['source_id'] == current_id and rel['target_id'] not in visited:
                visited.add(rel['target_id'])
                queue.append((rel['target_id'], path + [rel['target_id']]))
            
            # 检查入边（无向图）
            if rel['target_id'] == current_id and rel['source_id'] not in visited:
                visited.add(rel['source_id'])
                queue.append((rel['source_id'], path + [rel['source_id']]))
    
    return {'path': [], 'message': f'在{max_depth}步内未找到路径'}


def _build_path_result(path: List[int]) -> Dict[str, Any]:
    """构建路径结果"""
    # 获取路径上所有节点的关系
    relations = fetch_relations_for_nodes(path, limit=100)
    
    # 构建路径边
    path_edges = []
    for i in range(len(path) - 1):
        for rel in relations:
            if (rel['source_id'] == path[i] and rel['target_id'] == path[i+1]) or \
               (rel['target_id'] == path[i] and rel['source_id'] == path[i+1]):
                path_edges.append(rel)
                break
    
    return {
        'path': path,
        'path_edges': path_edges,
        'length': len(path) - 1,
        'message': f'找到长度为{len(path)-1}的路径'
    }
