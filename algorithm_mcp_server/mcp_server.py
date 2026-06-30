#!/usr/bin/env python3
"""Algorithm Teaching MCP Server

提供算法教学相关的工具和资源，供AI应用使用。

安装:
    pip install mcp fastmcp

运行:
    python mcp_server.py

或作为uvx包运行:
    uvx algorithm-teaching-mcp-server
"""

import sys
import os
from typing import Any, Dict, List, Optional

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from fastmcp import FastMCP
except ImportError:
    print("错误: 需要安装 fastmcp")
    print("运行: pip install fastmcp")
    sys.exit(1)

from app.rag import rag_ask
from app.db_source import fetch_algorithms, fetch_algorithm_by_id
from app.hybrid_retrieval import retrieve_rows
from app.store import get_chunks

# 创建MCP Server实例
mcp = FastMCP("Algorithm Teaching Server")

# ==================== Tools ====================

@mcp.tool()
def search_algorithm(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """搜索算法知识库
    
    参数:
        query: 搜索关键词（如"快速排序"、"二分查找"、"动态规划"）
        top_k: 返回结果数量（默认5）
        category: 算法类别过滤（可选）
    
    返回:
        匹配的算法chunk列表
    """
    # 获取所有chunks
    all_chunks = get_chunks()
    
    # 类别过滤
    if category:
        all_chunks = [c for c in all_chunks if c.get('chunk_type') == category]
    
    # 调用混合检索
    results = retrieve_rows(
        all_chunks,
        query,
        top_k=top_k,
        mode='hybrid',
        use_expansion=False,
        include_memory=False
    )
    
    # 格式化输出
    return [
        {
            'id': r['id'],
            'title': r['title'],
            'content': r['content'][:500] + '...' if len(r['content']) > 500 else r['content'],
            'algorithm_id': r.get('algorithm_id'),
            'score': r.get('score', 0),
            'chunk_type': r['chunk_type']
        }
        for r in results[:top_k]
    ]


@mcp.tool()
def get_algorithm_code(algorithm_name: str) -> Dict[str, Any]:
    """获取算法的完整代码实现
    
    参数:
        algorithm_name: 算法名称（如"快速排序"、"归并排序"）
    
    返回:
        算法代码和复杂度信息
    """
    algorithms = fetch_algorithms()
    
    # 查找匹配的算法
    algo = None
    for a in algorithms:
        if algorithm_name in a['algorithm_name'] or a['algorithm_name'] in algorithm_name:
            algo = a
            break
    
    if not algo:
        return {'error': f'算法 "{algorithm_name}" 未找到'}
    
    return {
        'algorithm_name': algo['algorithm_name'],
        'code': algo['code'],
        'step_text': algo.get('step_text', ''),
        'analysis_text': algo.get('analysis_text', '')
    }


@mcp.tool()
def explain_algorithm(algorithm_name: str, detail_level: str = "medium") -> Dict[str, str]:
    """解释算法的原理和实现
    
    参数:
        algorithm_name: 算法名称
        detail_level: 详细程度 (simple/medium/detailed)
    
    返回:
        详细解释
    """
    # 使用RAG生成解释
    question = f"详细解释{algorithm_name}算法"
    
    if detail_level == "simple":
        question = f"用简单的语言解释{algorithm_name}算法，适合初学者"
    elif detail_level == "detailed":
        question = f"非常详细地解释{algorithm_name}算法，包括证明和深入分析"
    
    result = rag_ask(
        question=question,
        top_k=5,
        retrieval_mode='hybrid',
        enable_tools=True
    )
    
    return {
        'algorithm_name': algorithm_name,
        'explanation': result['answer'],
        'references_count': len(result['references'])
    }


@mcp.tool()
def compare_algorithms(
    algorithm1: str,
    algorithm2: str
) -> Dict[str, Any]:
    """对比两个算法的异同
    
    参数:
        algorithm1: 第一个算法名称
        algorithm2: 第二个算法名称
    
    返回:
        详细的对比结果
    """
    # 生成对比问题
    question = f"{algorithm1}和{algorithm2}有什么区别？"
    
    result = rag_ask(
        question=question,
        top_k=8,
        retrieval_mode='hybrid',
        enable_tools=True
    )
    
    return {
        'algorithms': [algorithm1, algorithm2],
        'comparison': result['answer'],
        'references_count': len(result['references'])
    }


@mcp.tool()
def list_algorithms(category: Optional[str] = None) -> List[Dict[str, str]]:
    """列出所有算法
    
    参数:
        category: 按类别过滤（可选）
    
    返回:
        算法列表
    """
    algorithms = fetch_algorithms()
    
    result = [
        {
            'name': a['algorithm_name'],
            'id': a['algorithm_id'],
            'has_code': bool(a.get('code')),
            'has_steps': bool(a.get('step_text')),
            'has_analysis': bool(a.get('analysis_text'))
        }
        for a in algorithms
    ]
    
    return result[:50]  # 限制返回数量


# ==================== Resources ====================

@mcp.resource("algorithms://list")
def list_algorithms_resource() -> str:
    """算法列表资源"""
    algorithms = fetch_algorithms()
    
    output = "# 算法列表\n\n"
    output += f"共 {len(algorithms)} 个算法\n\n"
    
    for i, algo in enumerate(algorithms[:20], 1):
        output += f"{i}. {algo['algorithm_name']}\n"
    
    if len(algorithms) > 20:
        output += f"\n... 还有 {len(algorithms) - 20} 个算法"
    
    return output


@mcp.resource("algorithms://{name}")
def get_algorithm_resource(name: str) -> str:
    """获取单个算法的详细信息"""
    algorithms = fetch_algorithms()
    
    # 查找算法
    algo = None
    for a in algorithms:
        if name in a['algorithm_name'] or a['algorithm_name'].replace(' ', '_') == name:
            algo = a
            break
    
    if not algo:
        return f"# 错误\n\n算法 '{name}' 未找到"
    
    output = f"# {algo['algorithm_name']}\n\n"
    
    if algo.get('step_text'):
        output += "## 算法步骤\n\n"
        output += algo['step_text'] + "\n\n"
    
    if algo.get('analysis_text'):
        output += "## 复杂度分析\n\n"
        output += algo['analysis_text'] + "\n\n"
    
    if algo.get('code'):
        output += "## 代码实现\n\n```python\n"
        output += algo['code'][:1000]  # 限制长度
        if len(algo['code']) > 1000:
            output += "\n... (代码已截断)"
        output += "\n```\n"
    
    return output


# ==================== Prompts ====================

@mcp.prompt()
def explain_algorithm_prompt(algorithm_name: str) -> str:
    """生成"解释算法"的提示词模板"""
    return f"""我想学习{algorithm_name}算法，请帮我：

1. 解释算法的基本思想
2. 列出详细的算法步骤
3. 分析时间和空间复杂度
4. 说明典型的应用场景
5. 与类似算法对比优劣

请使用以下工具获取信息：
- search_algorithm("{algorithm_name}")
- get_algorithm_code("{algorithm_name}")
- explain_algorithm("{algorithm_name}")

用通俗易懂的语言解释，适合初学者理解。
"""


@mcp.prompt()
def compare_algorithms_prompt(algorithm1: str, algorithm2: str) -> str:
    """生成"对比算法"的提示词模板"""
    return f"""请详细对比{algorithm1}和{algorithm2}这两个算法：

对比维度：
1. 时间复杂度（最好/平均/最坏）
2. 空间复杂度
3. 稳定性
4. 适用场景
5. 实际性能
6. 实现难度

请使用工具：
- compare_algorithms("{algorithm1}", "{algorithm2}")
- get_algorithm_code("{algorithm1}")
- get_algorithm_code("{algorithm2}")

以表格形式总结主要差异。
"""


# ==================== Main ====================

if __name__ == "__main__":
    # 启动MCP Server
    print("启动 Algorithm Teaching MCP Server...")
    print("提供的工具:")
    print("  - search_algorithm: 搜索算法")
    print("  - get_algorithm_code: 获取代码")
    print("  - explain_algorithm: 解释算法")
    print("  - compare_algorithms: 对比算法")
    print("  - list_algorithms: 列出算法")
    print("")
    mcp.run()
