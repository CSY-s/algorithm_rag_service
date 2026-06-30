"""RAG 主流程模块。

负责构建知识库、执行检索、保存记忆和基础问答。"""


import hashlib
import json
import re
from typing import Optional

from .db_source import fetch_algorithms, fetch_algorithm_by_id
from .deepseek import deepseek_chat, expand_questions, extract_triples, synthesize_qa
from .hybrid_retrieval import retrieve_rows
from .store import get_chunks, insert_chunks, memory_hash_exists, reset_store
from .tools import build_tool_chunks
from .session import save_session_history, get_last_algorithm


def _guess_step(name: str, code: str) -> str:
    """当数据库缺少步骤说明时，从代码粗略推断步骤。"""
    lines = [x.strip() for x in code.splitlines() if x.strip()]
    return f"算法：{name}\n步骤线索（从代码提取）：\n" + '\n'.join(lines[:12])


def _meta(kind: str, **kwargs) -> str:
    """生成统一的 metadata JSON。"""
    payload = {'kind': kind}
    payload.update(kwargs)
    return json.dumps(payload, ensure_ascii=False)


def build_knowledge(with_synth: bool = True, limit: Optional[int] = None) -> dict:
    """从数据库构建静态知识库，并写入 chunk 表。"""
    rows = fetch_algorithms(limit=limit)
    reset_store()
    chunks = []

    for row in rows:
        aid = int(row['algorithm_id'])
        aname = row['algorithm_name']

        step = (row.get('step_text') or '').strip()
        if not step:
            step = _guess_step(aname, row['code'])

        chunks.append(
            {
                'algorithm_id': aid,
                'chunk_type': 'step',
                'title': f'{aname}-算法步骤',
                'content': step,
                'source': 'mysql_base',
                'metadata_json': _meta('step', algorithm_name=aname),
            }
        )

        analysis = (row.get('analysis_text') or '').strip()
        if analysis:
            chunks.append(
                {
                    'algorithm_id': aid,
                    'chunk_type': 'analysis',
                    'title': f'{aname}-算法分析',
                    'content': analysis,
                    'source': 'mysql_base',
                    'metadata_json': _meta('analysis', algorithm_name=aname),
                }
            )

        code_text = (row.get('code') or '').strip()
        if code_text:
            chunks.append(
                {
                    'algorithm_id': aid,
                    'chunk_type': 'code',
                    'title': f'{aname}-完整代码',
                    'content': code_text,
                    'source': 'mysql_base',
                    'metadata_json': _meta('code', algorithm_name=aname, full_code=True),
                }
            )

        for i, q in enumerate(row.get('question_docs', []), start=1):
            title = (q.get('title') or '').strip() or f'{aname}-题库问答{i}'
            ans = (q.get('ans') or '').strip()
            if not ans:
                continue
            chunks.append(
                {
                    'algorithm_id': aid,
                    'chunk_type': 'question_doc',
                    'title': title[:255],
                    'content': ans,
                    'source': 'mysql_question',
                    'metadata_json': _meta('question_doc', algorithm_name=aname),
                }
            )

        if with_synth:
            for i, qa in enumerate(synthesize_qa(aname, step, row['code']), start=1):
                chunks.append(
                    {
                        'algorithm_id': aid,
                        'chunk_type': 'qa',
                        'title': f'{aname}-合成问答{i}',
                        'content': f"Q: {qa['question']}\nA: {qa['answer']}",
                        'source': 'deepseek_synth',
                        'metadata_json': _meta('qa', algorithm_name=aname),
                    }
                )

    c = insert_chunks(chunks)
    return {'algorithms': len(rows), 'chunks': c}


def retrieve(
    question: str,
    top_k: int = 5,
    algorithm_id: int | None = None,
    mode: str = 'hybrid',
    use_expansion: bool = True,
    include_memory: bool = True,
) -> list[dict]:
    """从知识库中检索与问题最相关的片段。"""
    rows = get_chunks(algorithm_id=algorithm_id, include_memory=include_memory)
    return retrieve_rows(
        rows,
        question,
        top_k=top_k,
        mode=mode,
        use_expansion=use_expansion,
        include_memory=include_memory,
        expansion_fn=lambda q: expand_questions(q, max_n=3),
    )


def _save_memory(question: str, answer: str, algorithm_id: int | None = None):
    """把回答抽取成三元组后写成记忆 chunk。"""
    triples = extract_triples(answer, max_n=6)
    if not triples:
        return

    stable = sorted([f"{t['head']}|{t['relation']}|{t['tail']}" for t in triples])
    raw = f"{algorithm_id or 0}::" + '||'.join(stable)
    memory_hash = hashlib.md5(raw.encode('utf-8')).hexdigest()
    if memory_hash_exists(memory_hash, algorithm_id=algorithm_id):
        return

    content = '基于回答抽取的关系：\n' + '\n'.join([f"- {x}" for x in stable])
    insert_chunks(
        [
            {
                'algorithm_id': int(algorithm_id or 0),
                'chunk_type': 'memory',
                'title': '会话记忆-实体关系',
                'content': content,
                'source': 'agent_memory',
                'metadata_json': _meta('memory', memory_hash=memory_hash, from_question=question, triples=triples),
            }
        ]
    )


def _resolve_references(
    question: str, 
    algorithm_id: Optional[int],
    session_id: Optional[str]
) -> str:
    """消解指代词（"它"、"这个算法"等）
    
    示例:
        输入: "它的时间复杂度是多少?"
        上下文: algorithm_id=10 (快速排序)
        输出: "快速排序的时间复杂度是多少?"
    
    参数:
        question: 原始问题
        algorithm_id: 当前算法ID
        session_id: 会话ID
    
    返回:
        消解后的问题
    """
    # 如果没有algorithm_id，尝试从会话历史获取
    if not algorithm_id and session_id:
        algorithm_id = get_last_algorithm(session_id)
    
    if not algorithm_id:
        return question
    
    # 获取算法名称
    algo = fetch_algorithm_by_id(algorithm_id)
    if not algo:
        return question
    
    algo_name = algo['algorithm_name']
    
    # 替换指代词
    resolved = question
    
    # 替换常见指代词
    patterns = [
        (r'(?<![a-zA-Z])它(?![a-zA-Z])', algo_name),
        (r'这个算法', algo_name),
        (r'该算法', algo_name),
        (r'这种算法', algo_name),
        (r'上面的算法', algo_name),
        (r'当前算法', algo_name),
    ]
    
    for pattern, replacement in patterns:
        resolved = re.sub(pattern, replacement, resolved)
    
    return resolved


def rag_ask(
    question: str,
    top_k: int = 5,
    algorithm_id: int | None = None,
    session_id: str | None = None,
    retrieval_mode: str = 'hybrid',
    enable_tools: bool = True,
    enable_memory: bool = True,
    user_id: int | None = None,
) -> dict:
    """上下文感知的RAG问答流程
    
    参数:
        question: 用户问题
        top_k: 检索chunk数量
        algorithm_id: 限定算法范围（提供则只在该算法chunks中检索）
        session_id: 会话ID（用于追踪对话历史）
        retrieval_mode: 检索模式
        enable_tools: 是否启用工具增强
        enable_memory: 是否启用记忆机制
        user_id: 用户ID（可选，用于统计）
    
    返回:
        {
            'answer': 生成的答案,
            'references': 引用的chunk列表,
            'retrieval_mode': 使用的检索模式,
            'resolved_question': 消解后的问题（如果有指代消解）
        }
    """
    # 1. 处理指代消解
    resolved_question = _resolve_references(
        question, 
        algorithm_id=algorithm_id,
        session_id=session_id
    )
    
    # 2. 检索（限定algorithm_id范围）
    refs = retrieve(
        resolved_question,
        top_k=top_k,
        algorithm_id=algorithm_id,
        mode=retrieval_mode,
        use_expansion=True,
        include_memory=enable_memory,
    )
    
    if not refs:
        return {
            'answer': '暂无语料，请先调用/build构建。', 
            'references': [],
            'resolved_question': resolved_question
        }

    # 3. 工具增强
    tool_refs = build_tool_chunks(question, refs, algorithm_id=algorithm_id) if enable_tools else []
    final_refs = refs + tool_refs

    # 4. 构建上下文
    context = []
    for r in final_refs:
        context.append(
            f"[chunk_id={r.get('id')};algorithm_id={r.get('algorithm_id')};type={r.get('chunk_type')};source={r.get('source')}]\n{r.get('content')}"
        )

    # 5. LLM生成答案
    ans = deepseek_chat(
        [
            {
                'role': 'system',
                'content': '你是算法助教。优先依据检索片段回答，并在关键句后标注[chunk_id=xx]；若依据工具结果，请标注对应chunk_id。',
            },
            {'role': 'user', 'content': f"问题：{resolved_question}\n\n检索片段：\n\n" + '\n\n'.join(context)},
        ],
        temperature=0.2,
        max_tokens=1600,
    )

    # 6. 保存记忆
    if enable_memory:
        _save_memory(resolved_question, ans, algorithm_id=algorithm_id)
    
    # 7. 保存会话历史
    if session_id:
        save_session_history(
            session_id=session_id,
            user_id=user_id,
            algorithm_id=algorithm_id or 0,
            question=question,
            answer=ans,
            references=final_refs
        )

    return {
        'answer': ans, 
        'references': final_refs, 
        'retrieval_mode': retrieval_mode,
        'resolved_question': resolved_question if resolved_question != question else None
    }
