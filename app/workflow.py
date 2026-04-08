"""旧版规划工作流。

当前主入口已经切到 agent_workflow.py，这里保留旧实现作参考。"""


from __future__ import annotations

import json
from typing import Any

from .config import settings
from .deepseek import deepseek_chat
from .mcp_adapter import invoke_mcp_tool
from .rag import _save_memory, retrieve
from .tools import build_tool_chunks


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 对象。"""
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
    """旧版启发式规划器。"""
    q = (question or '').lower()
    before_tools: list[str] = []
    after_tools: list[str] = []

    if any(x in q for x in ['关系', '联系', '知识图谱', '依赖', '前置', '区别', '对比']):
        before_tools.append('knowledge_graph_lookup')
    if any(x in q for x in ['复杂度', '时间复杂度', '空间复杂度']):
        after_tools.append('complexity_extract')
    if any(x in q for x in ['今天', '最新', '最近', '官网', '联网']):
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
    """旧版 LLM 规划器。"""
    if not settings.planner_use_llm or not settings.deepseek_api_key:
        return None

    prompt = f"""
你是问答系统规划器。请只输出JSON对象。
目标：
1. 先规划；
2. 判断是否需要工具；
3. 若需要工具，优先通过MCP调用；
4. 若不需要工具，先尝试RAG检索；
5. 若RAG无有效内容，再直接由模型回答。

可用工具：
- knowledge_graph_lookup: 查询知识图谱节点和关系
- complexity_extract: 从已检索片段中抽取复杂度
- external_search: 需要联网/外部资料时使用

输出格式：
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
    """旧版统一规划入口。"""
    plan = _llm_plan(question)
    if plan:
        return plan
    return _heuristic_plan(question)


def _generate_answer_from_context(
    question: str,
    contexts: list[dict],
    use_citations: bool = True,
) -> str:
    """旧版基于上下文生成回答。"""
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


def _generate_direct_answer(question: str) -> str:
    """旧版无上下文时的直接回答。"""
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
    """旧版工具调用流程。"""
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
) -> dict[str, Any]:
    """旧版带规划的问答主流程。"""
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

    final_refs = rag_refs + tool_refs
    if final_refs:
        answer = _generate_answer_from_context(question, final_refs, use_citations=True)
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
    }
