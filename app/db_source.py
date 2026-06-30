"""数据库读取层。

负责从 MySQL 中读取算法、题库、知识图谱和评测样本等原始数据。"""


import re
from collections import defaultdict

import pymysql

from .config import settings


_CONN_ARGS = dict(
    host=settings.mysql_host,
    port=settings.mysql_port,
    user=settings.mysql_user,
    password=settings.mysql_password,
    database=settings.mysql_db,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)


def _get_conn():
    """创建一个新的 MySQL 连接。"""
    return pymysql.connect(**_CONN_ARGS)


def _clean_text(text: str) -> str:
    """清洗数据库中的 HTML、代码块和多余空白。"""
    if not text:
        return ''
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&emsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch_algorithms(limit: int | None = None) -> list[dict]:
    """读取算法基础信息，并把题库问答聚合到对应算法上。"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = 'SELECT id, name, code FROM algorithm_algorithm WHERE is_del=0 ORDER BY id'
            if limit and limit > 0:
                sql += f' LIMIT {int(limit)}'
            cur.execute(sql)
            algorithms = cur.fetchall()

            ids = [int(a['id']) for a in algorithms]
            qmap: dict[int, list[dict]] = defaultdict(list)
            if ids:
                placeholders = ','.join(['%s'] * len(ids))
                cur.execute(
                    f'''SELECT algorithm_id, title, ans
                        FROM algorithm_question
                        WHERE algorithm_id IN ({placeholders})
                        ORDER BY id''',
                    ids,
                )
                for row in cur.fetchall():
                    qmap[int(row['algorithm_id'])].append({
                        'title': (row.get('title') or '').strip(),
                        'ans': (row.get('ans') or '').strip(),
                    })

            out = []
            for a in algorithms:
                aid = int(a['id'])
                docs = qmap.get(aid, [])

                step_candidates = [
                    _clean_text(x['ans'])
                    for x in docs
                    if '步骤' in x['title'] and x.get('ans')
                ]
                analysis_candidates = [
                    _clean_text(x['ans'])
                    for x in docs
                    if '分析' in x['title'] and x.get('ans')
                ]

                out.append({
                    'algorithm_id': aid,
                    'algorithm_name': (a.get('name') or '').strip(),
                    'code': (a.get('code') or '').strip(),
                    'step_text': '\n'.join([x for x in step_candidates if x]),
                    'analysis_text': '\n'.join([x for x in analysis_candidates if x]),
                    'question_docs': docs,
                })
            return out
    finally:
        conn.close()


def fetch_algorithm_by_id(algorithm_id: int) -> dict | None:
    """根据算法ID获取算法信息
    
    参数:
        algorithm_id: 算法ID
    
    返回:
        算法信息字典，如果不存在返回None
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''SELECT id, name, code 
                   FROM algorithm_algorithm 
                   WHERE id=%s AND is_del=0 
                   LIMIT 1''',
                (int(algorithm_id),)
            )
            row = cur.fetchone()
            if not row:
                return None
            
            return {
                'algorithm_id': int(row['id']),
                'algorithm_name': (row.get('name') or '').strip(),
                'code': (row.get('code') or '').strip(),
            }
    finally:
        conn.close()


def search_knowledge_nodes(query: str, limit: int = 6) -> list[dict]:
    """按关键词模糊搜索知识图谱节点。"""
    query = (query or '').strip()
    if not query:
        return []
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            like = f'%{query}%'
            cur.execute(
                '''SELECT id, name, COALESCE(type, '') AS type, COALESCE(description, '') AS description
                   FROM knowledge_node
                   WHERE name LIKE %s OR description LIKE %s
                   ORDER BY id
                   LIMIT %s''',
                (like, like, max(1, int(limit))),
            )
            rows = cur.fetchall()
            for r in rows:
                r['description'] = _clean_text(r.get('description') or '')[:240]
            return rows
    finally:
        conn.close()


def fetch_relations_for_nodes(node_ids: list[int], limit: int = 16) -> list[dict]:
    """根据节点 id 查询相关关系边。"""
    if not node_ids:
        return []
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(node_ids))
            cur.execute(
                f'''SELECT
                        r.source_id,
                        s.name AS source_name,
                        r.relation_type,
                        r.target_id,
                        t.name AS target_name
                    FROM knowledge_relation r
                    JOIN knowledge_node s ON s.id = r.source_id
                    JOIN knowledge_node t ON t.id = r.target_id
                    WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
                    ORDER BY r.id
                    LIMIT %s''',
                tuple(node_ids + node_ids + [max(1, int(limit))]),
            )
            return cur.fetchall()
    finally:
        conn.close()


def _make_eval_question(algorithm_name: str, title: str) -> str:
    """把题库标题改写成更适合自动评测的自然语言问题。"""
    t = (title or '').strip()
    if '步骤' in t:
        return f'{algorithm_name}的核心步骤是什么？'
    if '分析' in t:
        return f'{algorithm_name}的复杂度和实现要点是什么？'
    if '代码' in t:
        return f'{algorithm_name}的关键代码思路是什么？'
    return t or f'{algorithm_name}相关知识点是什么？'


def fetch_eval_samples(limit: int = 12, algorithm_id: int | None = None) -> list[dict]:
    """生成评测样本：问题 + 参考答案。"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            params: list = []
            sql = '''SELECT q.id, q.algorithm_id, q.title, q.ans, a.name AS algorithm_name
                     FROM algorithm_question q
                     JOIN algorithm_algorithm a ON a.id = q.algorithm_id
                     WHERE COALESCE(q.ans, '') <> '' '''
            sql += " AND q.title NOT LIKE '%%代码%%' "
            if algorithm_id is not None:
                sql += ' AND q.algorithm_id=%s '
                params.append(int(algorithm_id))
            sql += ' ORDER BY q.id LIMIT %s'
            params.append(max(1, int(limit)))
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

            out = []
            for r in rows:
                answer = _clean_text(r.get('ans') or '')
                if len(answer) < 10:
                    continue
                algorithm_name = (r.get('algorithm_name') or '').strip()
                out.append(
                    {
                        'sample_id': int(r['id']),
                        'algorithm_id': int(r['algorithm_id']),
                        'algorithm_name': algorithm_name,
                        'question': _make_eval_question(algorithm_name, r.get('title') or ''),
                        'reference_answer': answer[:1800],
                        'source_title': (r.get('title') or '').strip(),
                    }
                )
            return out
    finally:
        conn.close()



def fetch_all_knowledge_nodes(limit: int | None = None) -> list[dict]:
    """获取所有知识图谱节点
    
    参数:
        limit: 限制返回数量
    
    返回:
        节点列表
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = 'SELECT id, name, COALESCE(type, "") AS type, COALESCE(description, "") AS description FROM knowledge_node ORDER BY id'
            if limit:
                sql += f' LIMIT {int(limit)}'
            cur.execute(sql)
            rows = cur.fetchall()
            for r in rows:
                r['description'] = _clean_text(r.get('description') or '')[:240]
            return rows
    finally:
        conn.close()


def fetch_all_knowledge_relations(node_ids: list[int] | None = None) -> list[dict]:
    """获取所有知识图谱关系（可选过滤节点）
    
    参数:
        node_ids: 可选，只返回这些节点相关的关系
    
    返回:
        关系列表
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if node_ids:
                placeholders = ','.join(['%s'] * len(node_ids))
                sql = f'''
                SELECT r.id, r.source_id, s.name AS source_name,
                       r.relation_type,
                       r.target_id, t.name AS target_name
                FROM knowledge_relation r
                JOIN knowledge_node s ON s.id = r.source_id
                JOIN knowledge_node t ON t.id = r.target_id
                WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
                ORDER BY r.id
                '''
                cur.execute(sql, tuple(node_ids + node_ids))
            else:
                sql = '''
                SELECT r.id, r.source_id, s.name AS source_name,
                       r.relation_type,
                       r.target_id, t.name AS target_name
                FROM knowledge_relation r
                JOIN knowledge_node s ON s.id = r.source_id
                JOIN knowledge_node t ON t.id = r.target_id
                ORDER BY r.id
                '''
                cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()
