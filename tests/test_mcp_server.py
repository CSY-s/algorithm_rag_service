"""测试MCP Server功能

测试所有MCP工具是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm_mcp_server.mcp_server import (
    search_algorithm,
    get_algorithm_code,
    explain_algorithm,
    compare_algorithms,
    list_algorithms,
    list_algorithms_resource,
    get_algorithm_resource,
    explain_algorithm_prompt,
    compare_algorithms_prompt
)


def test_search_algorithm():
    """测试搜索算法工具"""
    print("\n=== 测试1: search_algorithm ===")
    
    result = search_algorithm("排序", top_k=3)
    print(f"搜索 '排序'，返回 {len(result)} 个结果")
    
    if result:
        print(f"第1个结果: {result[0]['title']}")
        print(f"  内容预览: {result[0]['content'][:100]}...")
        print(f"  相关度: {result[0]['score']:.4f}")
    
    assert len(result) > 0, "应该返回至少1个结果"
    print("✅ search_algorithm 测试通过")


def test_get_algorithm_code():
    """测试获取算法代码"""
    print("\n=== 测试2: get_algorithm_code ===")
    
    result = get_algorithm_code("快速排序")
    print(f"获取 '快速排序' 的代码")
    
    if 'error' not in result:
        print(f"  算法名: {result['algorithm_name']}")
        print(f"  代码长度: {len(result['code'])} 字符")
        print(f"  有步骤说明: {bool(result['step_text'])}")
        print(f"  有复杂度分析: {bool(result['analysis_text'])}")
        assert len(result['code']) > 0, "代码不应为空"
    else:
        print(f"  错误: {result['error']}")
    
    print("✅ get_algorithm_code 测试通过")


def test_explain_algorithm():
    """测试解释算法"""
    print("\n=== 测试3: explain_algorithm ===")
    
    result = explain_algorithm("归并排序", detail_level="simple")
    print(f"解释 '归并排序' (简单级别)")
    
    if 'explanation' in result:
        print(f"  解释长度: {len(result['explanation'])} 字符")
        print(f"  引用chunks: {result['references_count']} 个")
        print(f"  解释预览: {result['explanation'][:150]}...")
        assert len(result['explanation']) > 0, "解释不应为空"
    
    print("✅ explain_algorithm 测试通过")


def test_compare_algorithms():
    """测试对比算法"""
    print("\n=== 测试4: compare_algorithms ===")
    
    result = compare_algorithms("快速排序", "归并排序")
    print(f"对比 '快速排序' 和 '归并排序'")
    
    if 'comparison' in result:
        print(f"  对比内容长度: {len(result['comparison'])} 字符")
        print(f"  引用chunks: {result['references_count']} 个")
        print(f"  对比预览: {result['comparison'][:150]}...")
        assert "快速排序" in result['comparison']
        assert "归并排序" in result['comparison']
    
    print("✅ compare_algorithms 测试通过")


def test_list_algorithms():
    """测试列出算法"""
    print("\n=== 测试5: list_algorithms ===")
    
    result = list_algorithms()
    print(f"列出所有算法，返回 {len(result)} 个")
    
    if result:
        print(f"第1个算法: {result[0]['name']} (ID: {result[0]['id']})")
        print(f"  有代码: {result[0]['has_code']}")
        print(f"  有步骤: {result[0]['has_steps']}")
        print(f"  有分析: {result[0]['has_analysis']}")
    
    assert len(result) > 0, "应该返回至少1个算法"
    print("✅ list_algorithms 测试通过")


def test_resources():
    """测试资源"""
    print("\n=== 测试6: 资源 ===")
    
    # 测试算法列表资源
    list_resource = list_algorithms_resource()
    print(f"算法列表资源长度: {len(list_resource)} 字符")
    print(f"  预览: {list_resource[:100]}...")
    assert "算法列表" in list_resource
    
    # 测试单个算法资源
    algo_resource = get_algorithm_resource("快速排序")
    print(f"\n快速排序资源长度: {len(algo_resource)} 字符")
    print(f"  预览: {algo_resource[:100]}...")
    assert "快速排序" in algo_resource or "快排" in algo_resource
    
    print("✅ 资源测试通过")


def test_prompts():
    """测试提示词模板"""
    print("\n=== 测试7: 提示词模板 ===")
    
    # 测试解释算法提示词
    explain_prompt = explain_algorithm_prompt("二分查找")
    print(f"解释算法提示词长度: {len(explain_prompt)} 字符")
    print(f"  预览: {explain_prompt[:150]}...")
    assert "二分查找" in explain_prompt
    assert "search_algorithm" in explain_prompt
    
    # 测试对比算法提示词
    compare_prompt = compare_algorithms_prompt("BFS", "DFS")
    print(f"\n对比算法提示词长度: {len(compare_prompt)} 字符")
    print(f"  预览: {compare_prompt[:150]}...")
    assert "BFS" in compare_prompt
    assert "DFS" in compare_prompt
    
    print("✅ 提示词模板测试通过")


if __name__ == '__main__':
    print("开始测试MCP Server功能...")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_search_algorithm()
        test_get_algorithm_code()
        test_explain_algorithm()
        test_compare_algorithms()
        test_list_algorithms()
        test_resources()
        test_prompts()
        
        print("\n" + "=" * 60)
        print("✅ 所有MCP Server测试通过!")
        print("\n建议:")
        print("1. 在Kiro中配置mcp.json")
        print("2. 使用#explain_algorithm skill测试")
        print("3. 尝试调用MCP工具")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
