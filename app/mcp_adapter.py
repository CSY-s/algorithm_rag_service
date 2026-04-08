"""MCP 工具适配层。

优先尝试远程 MCP，失败后回退到本地工具。"""


from __future__ import annotations

from typing import Any

import requests

from .config import settings
from .tools import build_complexity_tool_chunk, build_kg_tool_chunk


def _remote_mcp_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """尝试调用远程 MCP 服务。"""
    if not settings.mcp_enabled or not settings.mcp_base_url:
        return {'success': False, 'error': 'MCP未启用或未配置MCP_BASE_URL'}

    payload = {'tool': tool_name, 'arguments': arguments}
    resp = requests.post(settings.mcp_base_url, json=payload, timeout=settings.mcp_timeout)
    resp.raise_for_status()
    data = resp.json()
    return {
        'success': bool(data.get('success', True)),
        'result': data.get('result'),
        'raw': data,
    }


def _local_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """当远程 MCP 不可用时，回退到本地工具实现。"""
    if tool_name == 'knowledge_graph_lookup':
        chunk = build_kg_tool_chunk(
            arguments.get('question', ''),
            algorithm_id=arguments.get('algorithm_id'),
        )
        return {'success': chunk is not None, 'result': chunk, 'source': 'local_tool'}

    if tool_name == 'complexity_extract':
        chunk = build_complexity_tool_chunk(
            arguments.get('refs', []),
            algorithm_id=arguments.get('algorithm_id'),
        )
        return {'success': chunk is not None, 'result': chunk, 'source': 'local_tool'}

    return {'success': False, 'error': f'未知工具: {tool_name}'}


def invoke_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """统一的工具调用入口：先远程 MCP，再本地兜底。"""
    remote_error = None
    if settings.mcp_enabled and settings.mcp_base_url:
        try:
            remote = _remote_mcp_call(tool_name, arguments)
            if remote.get('success'):
                result = remote.get('result')
                if isinstance(result, dict):
                    result.setdefault('source', 'mcp_remote')
                return {
                    'success': True,
                    'result': result,
                    'source': 'mcp_remote',
                }
            remote_error = remote.get('error') or '远程MCP调用失败'
        except Exception as e:
            remote_error = str(e)

    local = _local_tool_call(tool_name, arguments)
    if local.get('success'):
        return local

    return {
        'success': False,
        'error': remote_error or local.get('error') or '工具调用失败',
        'source': 'mcp_fallback',
    }
