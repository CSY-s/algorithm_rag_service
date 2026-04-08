"""规划式智能体工作流。

负责先规划，再决定是否调用工具、做 RAG 检索或直接回答。"""


from __future__ import annotations

import json
from typing import Any

from .config import settings
from .deepseek import deepseek_chat
from .memory_graph import build_memory_context_chunk
from .mcp_adapter import invoke_mcp_tool
from .rag import _save_memory, retrieve
from .tools import build_tool_chunks


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """从模型输出中尽量提取 JSON 对象。"""
    if not raw:
        return None
    st = raw.find('{')
    ed = raw.rfind('}')
    if st == -1 or ed == -1 or ed <= st:
        return None
    try:
        obj = json.loads(raw[st : ed + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _heuristic_plan(question: str) -> dict[str, Any]:
    """当 LLM 规划不可用时，使用关键词规则做启发式规划。"""
    q = (question or '').lower()
    before_tools: list[str] = []
    after_tools: list[str] = []

    kg_keywords = ['关系', '联系', '知识图谱', '依赖', '前置', '区别', '对比']
    complexity_keywords = ['复杂度', '时间复杂度', '空间复杂度']
    web_keywords = ['今天', '最新', '最近', '官网', '联网']

    if any(x in q for x in kg_keywords):
        before_tools.append('knowledge_graph_lookup')
    if any(x in q for x in complexity_keywords):
        after_tools.append('complexity_extract')
    if any(x in q for x in web_keywords):
        before_tools.append('external_search')

    return {
        'planner': 'heuristic',
        'need_tools_before_rag': bool(before_tools),
        'need_tools_after_rag': bool(after_tools),
        'tools_before_rag': before_tools,
        'tools_after_rag': after_tools,
        'try_rag': True,
        'fallback_direct_llm': True,
        'reason': '基于关键词启发式规划',
    }


def _llm_plan(question: str) -> dict[str, Any] | None:
    """让大模型先判断是否需要工具、RAG 和兜底直答。"""
    if not settings.planner_use_llm or not settings.deepseek_api_key:
        return None

    prompt = f"""
你是问答系统规划器，只输出JSON对象。

执行原则：
1. 先规划。
2. 先判断是否需要工具。
3. 如果需要工具，优先通过MCP调用。
4. 如果不需要工具，先尝试RAG检索。
5. 如果RAG没有有效上下文，再直接由模型回答。
6. 如果工具或RAG某一步失败，优先尝试另一种能力，最后才直接回答。

可用工具：
- knowledge_graph_lookup: 查询知识图谱节点和关系
- complexity_extract: 从已检索片段中抽取复杂度
- external_search: 需要外部/联网信息时使用

返回格式：
{{
  "need_tools_before_rag": true,
  "need_tools_after_rag": false,
  "tools_before_rag": ["knowledge_graph_lookup"],
  "tools_after_rag": [],
  "try_rag": true,
  "fallback_direct_llm": true,
  "reason": "..."
}}

用户问题：{question}
"""
    try:
        raw = deepseek_chat(
            [
                {'role': 'system', 'content': '你是规划器，只输出JSON对象。'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        obj = _extract_json_object(raw)
        if not obj:
            return None
        obj['planner'] = 'llm'
        return obj
    except Exception:
        return None


def make_plan(question: str) -> dict[str, Any]:
    """统一规划入口：优先 LLM，失败后回退到启发式规划。"""
    plan = _llm_plan(question)
    if plan:
        return plan
    return _heuristic_plan(question)


def _generate_answer_from_context(question: str, contexts: list[dict], use_citations: bool = True) -> str:
    """仅基于给定上下文生成回答。"""
    if not contexts:
        return ''

    content = []
    for item in contexts:
        content.append(
            f"[chunk_id={item.get('id')};algorithm_id={item.get('algorithm_id')};type={item.get('chunk_type')};source={item.get('source')}]\n{item.get('content')}"
        )

    system_prompt = '你是算法助教。优先依据给定上下文回答。'
    if use_citations:
        system_prompt += '请在关键句后标注[chunk_id=xx]。'

    return deepseek_chat(
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"问题：{question}\n\n上下文：\n\n" + '\n\n'.join(content)},
        ],
        temperature=0.2,
        max_tokens=1600,
    )


def _build_mixed_knowledge(question: str, rag_refs: list[dict], tool_refs: list[dict], current_state: str = '') -> dict[str, str]:
    """把检索知识、工具结果、记忆和当前状态组织成混合知识。"""
    rag_text = '\n\n'.join(
        f"[{item.get('chunk_type')}#{item.get('id')}]\n{item.get('content')}" for item in rag_refs
    )
    tool_text = '\n\n'.join(
        f"[{item.get('chunk_type')}#{item.get('id')}]\n{item.get('content')}" for item in tool_refs
    )
    mixed = [
        f"用户问题：{question}",
        '选择知识：',
        rag_text or '(无)',
        '选择工具/记忆：',
        tool_text or '(无)',
        '当前状态S：',
        current_state or '(无)',
    ]
    return {
        'rag_knowledge': rag_text,
        'tool_knowledge': tool_text,
        'current_state': current_state or '',
        'mixed_context': '\n\n'.join(mixed),
    }


def _generate_reasoned_answer(question: str, mixed_knowledge: dict[str, str], use_citations: bool = True) -> str:
    """基于混合知识进行链式推理后输出答案。"""
    system_prompt = '你是算法教学智能体。请先基于混合知识进行分步推理，再给出最终回答。'
    if use_citations:
        system_prompt += '在关键句后标注[chunk_id=xx]。'
    prompt = f"""
请根据以下混合知识进行回答：

{mixed_knowledge['mixed_context']}

要求：
1. 先整理关键依据；
2. 再进行思维链式推理；
3. 最后给出清晰、结构化的教学回答。
"""
    return deepseek_chat(
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.2,
        max_tokens=1800,
    )


def _generate_direct_answer(question: str) -> str:
    """当没有工具或 RAG 上下文时，直接用模型回答。"""
    return deepseek_chat(
        [
            {
                'role': 'system',
                'content': '你是算法助教。当没有工具和RAG上下文时，请直接给出尽量可靠、清晰的回答，并明确说明这是基于模型通用知识的回答。',
            },
            {'role': 'user', 'content': question},
        ],
        temperature=0.3,
        max_tokens=1400,
    )


def _run_tools(tool_names: list[str], question: str, algorithm_id: int | None, refs: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """按规划依次执行工具，并记录执行轨迹。"""
    outputs: list[dict] = []
    trace: list[dict] = []

    for tool_name in tool_names:
        result = invoke_mcp_tool(
            tool_name,
            {
                'question': question,
                'algorithm_id': algorithm_id,
                'refs': refs or [],
            },
        )
        trace.append(
            {
                'stage': 'tool',
                'tool_name': tool_name,
                'success': bool(result.get('success')),
                'source': result.get('source'),
                'error': result.get('error'),
            }
        )
        if result.get('success') and isinstance(result.get('result'), dict):
            outputs.append(result['result'])

    return outputs, trace


def answer_with_plan(
    question: str,
    top_k: int = 5,
    algorithm_id: int | None = None,
    retrieval_mode: str = 'hybrid',
    enable_tools: bool = True,
    enable_memory: bool = True,
    enable_planning: bool = True,
    enable_mcp: bool = True,
    current_state: str = '',
) -> dict[str, Any]:
    """当前系统的主问答流程：先规划，再按规划逐步执行。"""
    plan = make_plan(question) if enable_planning else _heuristic_plan(question)
    trace: list[dict] = [{'stage': 'plan', 'plan': plan}]
    tool_refs: list[dict] = []
    rag_refs: list[dict] = []

    before_tools = plan.get('tools_before_rag', []) if enable_tools and enable_mcp else []
    after_tools = plan.get('tools_after_rag', []) if enable_tools and enable_mcp else []

    if plan.get('need_tools_before_rag') and before_tools:
        tool_refs, tool_trace = _run_tools(before_tools, question, algorithm_id)
        trace.extend(tool_trace)

    if plan.get('try_rag', True):
        rag_refs = retrieve(
            question,
            top_k=top_k,
            algorithm_id=algorithm_id,
            mode=retrieval_mode,
            use_expansion=True,
            include_memory=enable_memory,
        )
        trace.append({'stage': 'rag', 'retrieval_mode': retrieval_mode, 'hits': len(rag_refs)})

    if plan.get('need_tools_after_rag') and after_tools:
        post_tool_refs, tool_trace = _run_tools(after_tools, question, algorithm_id, refs=rag_refs)
        tool_refs.extend(post_tool_refs)
        trace.extend(tool_trace)

    if enable_tools:
        fallback_tools = build_tool_chunks(question, rag_refs, algorithm_id=algorithm_id)
        existing_keys = {(x.get('title'), x.get('source')) for x in tool_refs}
        for item in fallback_tools:
            key = (item.get('title'), item.get('source'))
            if key not in existing_keys:
                tool_refs.append(item)

    memory_chunk = build_memory_context_chunk(question, algorithm_id=algorithm_id, top_k=max(4, top_k))
    if memory_chunk:
        tool_refs.append(memory_chunk)
        trace.append(
            {
                'stage': 'memory_graph',
                'selected': len(memory_chunk.get('selected_memory', [])),
                'attached': True,
            }
        )

    final_refs = rag_refs + tool_refs
    mixed_knowledge = _build_mixed_knowledge(question, rag_refs, tool_refs, current_state=current_state)
    if final_refs:
        answer = _generate_reasoned_answer(question, mixed_knowledge, use_citations=True)
        answer_mode = 'tool_rag' if tool_refs and rag_refs else ('tool_only' if tool_refs else 'rag')
    else:
        if enable_tools and not before_tools and not after_tools:
            retry_tools, tool_trace = _run_tools(['knowledge_graph_lookup'], question, algorithm_id)
            tool_refs.extend(retry_tools)
            trace.extend(tool_trace)
            if tool_refs:
                final_refs = tool_refs
                answer = _generate_answer_from_context(question, final_refs, use_citations=True)
                answer_mode = 'tool_only'
            else:
                answer = _generate_direct_answer(question)
                answer_mode = 'direct_llm'
        else:
            answer = _generate_direct_answer(question)
            answer_mode = 'direct_llm'

    if enable_memory:
        try:
            _save_memory(question, answer, algorithm_id=algorithm_id)
            trace.append({'stage': 'memory', 'saved': True})
        except Exception as e:
            trace.append({'stage': 'memory', 'saved': False, 'error': str(e)})

    return {
        'answer': answer,
        'plan': plan,
        'execution_trace': trace,
        'references': final_refs,
        'tool_references': tool_refs,
        'rag_references': rag_refs,
        'retrieval_mode': retrieval_mode,
        'answer_mode': answer_mode,
        'mixed_knowledge': mixed_knowledge,
    }
