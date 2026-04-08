"""知识库存储层。

负责把 chunk 写入 algorithm_rag_chunk，并按条件读出。"""


import datetime
import json

import pymysql

from .config import settings


_BUILDER = 'algorithm_rag_service'
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
    """创建知识库存储所需的数据库连接。"""
    return pymysql.connect(**_CONN_ARGS)


def _with_builder_metadata(raw: str | None) -> str:
    """给 metadata 补上 builder 标记，便于区分本项目写入的数据。"""
    payload = {}
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                payload.update(obj)
        except Exception:
            payload = {'raw': str(raw)}
    payload.setdefault('builder', _BUILDER)
    return json.dumps(payload, ensure_ascii=False)


def reset_store():
    """删除本项目之前构建出来的知识库内容。"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM algorithm_rag_chunk
                   WHERE source IN ('mysql_base', 'mysql_question', 'deepseek_synth', 'agent_memory', 'tool_kg')
                      OR metadata_json LIKE %s""",
                (f'%"builder": "{_BUILDER}"%',),
            )
    finally:
        conn.close()


def insert_chunks(rows: list[dict]) -> int:
    """批量写入 chunk。"""
    if not rows:
        return 0

    now = datetime.datetime.now()
    data = []
    for r in rows:
        data.append(
            (
                int(r.get('algorithm_id', 0) or 0),
                str(r.get('chunk_type') or 'text'),
                str(r.get('title') or '')[:255],
                str(r.get('content') or ''),
                str(r.get('source') or 'mysql_base')[:64],
                _with_builder_metadata(r.get('metadata_json')),
                now,
                now,
            )
        )

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                '''INSERT INTO algorithm_rag_chunk
                   (algorithm_id, chunk_type, title, content, source, metadata_json, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                data,
            )
        return len(rows)
    finally:
        conn.close()


def get_chunks(algorithm_id: int | None = None, include_memory: bool = True) -> list[dict]:
    """按条件读取 chunk，可选择是否包含 memory。"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if algorithm_id is None:
                cur.execute(
                    '''SELECT id, algorithm_id, chunk_type, title, content, source, metadata_json
                       FROM algorithm_rag_chunk
                       WHERE source IN ('mysql_base', 'mysql_question', 'deepseek_synth', 'agent_memory', 'tool_kg')
                          OR metadata_json LIKE %s
                       ORDER BY id''',
                    (f'%"builder": "{_BUILDER}"%',),
                )
            elif include_memory:
                cur.execute(
                    '''SELECT id, algorithm_id, chunk_type, title, content, source, metadata_json
                       FROM algorithm_rag_chunk
                       WHERE (algorithm_id=%s OR (chunk_type='memory' AND algorithm_id IN (0, %s)))
                         AND (source IN ('mysql_base', 'mysql_question', 'deepseek_synth', 'agent_memory', 'tool_kg')
                              OR metadata_json LIKE %s)
                       ORDER BY id''',
                    (int(algorithm_id), int(algorithm_id), f'%"builder": "{_BUILDER}"%'),
                )
            else:
                cur.execute(
                    '''SELECT id, algorithm_id, chunk_type, title, content, source, metadata_json
                       FROM algorithm_rag_chunk
                       WHERE algorithm_id=%s
                         AND (source IN ('mysql_base', 'mysql_question', 'deepseek_synth', 'agent_memory', 'tool_kg')
                              OR metadata_json LIKE %s)
                       ORDER BY id''',
                    (int(algorithm_id), f'%"builder": "{_BUILDER}"%'),
                )
            return cur.fetchall()
    finally:
        conn.close()


def memory_hash_exists(memory_hash: str, algorithm_id: int | None = None) -> bool:
    """检查同一份记忆是否已经写入过。"""
    if not memory_hash:
        return False
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            like = f'%"memory_hash": "{memory_hash}"%'
            if algorithm_id is None:
                cur.execute(
                    '''SELECT 1 FROM algorithm_rag_chunk
                       WHERE chunk_type='memory' AND metadata_json LIKE %s LIMIT 1''',
                    (like,),
                )
            else:
                cur.execute(
                    '''SELECT 1 FROM algorithm_rag_chunk
                       WHERE chunk_type='memory' AND algorithm_id IN (0, %s) AND metadata_json LIKE %s LIMIT 1''',
                    (int(algorithm_id), like),
                )
            return cur.fetchone() is not None
    finally:
        conn.close()
