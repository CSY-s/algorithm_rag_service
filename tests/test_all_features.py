"""测试所有新功能

测试内容:
1. 上下文感知RAG
2. 会话管理
3. MCP Server
4. 集成场景
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import rag_ask
from app.session import (
    save_session_history,
    get_session_history,
    get_session_summary,
    clear_session
)
from algorithm_mcp_server.mcp_server import (
    search_algorithm,
    get_algorithm_code,
    explain_algorithm
)


def test_integration_scenario():
    """测试完整集成场景"""
    print("\n=== 集成场景测试 ===")
    print("模拟用户在算法详情页的完整交互流程\n")
    
    # 场景：用户访问"快速排序"页面（假设ID=10）
    algorithm_id = 10
    session_id = "integration_test_001"
    clear_session(session_id)
    
    print("【场景】用户访问 /logicalContent/10 (快速排序)")
    print("")
    
    # 第1轮：询问基本概念
    print("第1轮对话：")
    print("  用户: 这个算法的基本思想是什么?")
    
    result1 = rag_ask(
        question="这个算法的基本思想是什么?",
        algorithm_id=algorithm_id,
        session_id=session_id,
        top_k=3
    )
    
    print(f"  系统消解: {result1.get('resolved_question', '(无需消解)')}")
    print(f"  系统回答: {result1['answer'][:100]}...")
    print("")
    
    # 第2轮：使用指代词提问
    print("第2轮对话：")
    print("  用户: 它的时间复杂度是多少?")
    
    result2 = rag_ask(
        question="它的时间复杂度是多少?",
        algorithm_id=algorithm_id,
        session_id=session_id,
        top_k=3
    )
    
    print(f"  系统消解: {result2.get('resolved_question')}")
    print(f"  系统回答: {result2['answer'][:100]}...")
    print("")
    
    # 第3轮：深入提问
    print("第3轮对话：")
    print("  用户: 最坏情况是什么?")
    
    result3 = rag_ask(
        question="最坏情况是什么?",
        algorithm_id=algorithm_id,
        session_id=session_id,
        top_k=3
    )
    
    print(f"  系统回答: {result3['answer'][:100]}...")
    print("")
    
    # 查看会话历史
    history = get_session_history(session_id)
    print(f"会话历史: 共 {len(history)} 轮对话")
    
    # 查看会话摘要
    summary = get_session_summary(session_id)
    print(f"会话摘要:")
    print(f"  总消息数: {summary['total_messages']}")
    print(f"  讨论的算法: {summary['algorithms_discussed']}")
    print("")
    
    # 验证
    assert len(history) == 3, "应该有3轮对话"
    assert algorithm_id in summary['algorithms_discussed'], "应该记录了算法ID"
    assert "快速排序" in result2.get('resolved_question', ''), "应该消解了指代"
    
    print("✅ 集成场景测试通过")


def test_mcp_integration():
    """测试MCP工具集成"""
    print("\n=== MCP工具集成测试 ===")
    
    # 场景：用户想对比两个算法
    print("【场景】用户想对比快排和归并\n")
    
    # 1. 先用MCP搜索相关算法
    print("步骤1: 使用MCP搜索工具")
    search_results = search_algorithm("排序", top_k=3)
    print(f"  搜索到 {len(search_results)} 个相关算法")
    if search_results:
        print(f"  第1个: {search_results[0]['title']}")
    print("")
    
    # 2. 用MCP获取代码
    print("步骤2: 使用MCP获取代码工具")
    code_result = get_algorithm_code("快速排序")
    if 'error' not in code_result:
        print(f"  获取到 {code_result['algorithm_name']} 的代码")
        print(f"  代码长度: {len(code_result['code'])} 字符")
    print("")
    
    # 3. 用MCP解释算法
    print("步骤3: 使用MCP解释工具")
    explain_result = explain_algorithm("快速排序", detail_level="simple")
    print(f"  解释长度: {len(explain_result['explanation'])} 字符")
    print(f"  引用了 {explain_result['references_count']} 个chunks")
    print("")
    
    # 4. 结合RAG进行对话
    print("步骤4: 结合RAG进行深度对话")
    rag_result = rag_ask(
        question="快速排序和归并排序哪个更好?",
        top_k=5,
        retrieval_mode='hybrid'
    )
    print(f"  RAG回答: {rag_result['answer'][:150]}...")
    print("")
    
    assert len(search_results) > 0, "搜索应该返回结果"
    assert 'error' not in code_result, "获取代码不应出错"
    assert len(explain_result['explanation']) > 0, "解释不应为空"
    
    print("✅ MCP工具集成测试通过")


def test_cross_algorithm_comparison():
    """测试跨算法对比场景"""
    print("\n=== 跨算法对比测试 ===")
    
    session_id = "comparison_test_001"
    clear_session(session_id)
    
    print("【场景】用户想学习多个排序算法\n")
    
    # 1. 学习快排
    print("步骤1: 学习快速排序")
    result1 = rag_ask(
        question="快速排序是怎么工作的?",
        algorithm_id=10,
        session_id=session_id
    )
    print(f"  回答长度: {len(result1['answer'])} 字符")
    print("")
    
    # 2. 学习归并
    print("步骤2: 学习归并排序")
    result2 = rag_ask(
        question="归并排序是怎么工作的?",
        algorithm_id=15,  # 假设15是归并排序
        session_id=session_id
    )
    print(f"  回答长度: {len(result2['answer'])} 字符")
    print("")
    
    # 3. 对比两者
    print("步骤3: 对比两个算法")
    result3 = rag_ask(
        question="这两个算法有什么区别?",
        session_id=session_id
    )
    print(f"  回答长度: {len(result3['answer'])} 字符")
    print("")
    
    # 查看学习路径
    summary = get_session_summary(session_id)
    print(f"学习路径:")
    print(f"  讨论的算法: {summary['algorithms_discussed']}")
    print(f"  总对话轮次: {summary['total_messages']}")
    print("")
    
    assert len(summary['algorithms_discussed']) >= 2, "应该讨论了至少2个算法"
    
    print("✅ 跨算法对比测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("🚀 开始测试所有新功能")
    print("=" * 70)
    
    try:
        # 1. 集成场景测试
        test_integration_scenario()
        
        # 2. MCP集成测试
        test_mcp_integration()
        
        # 3. 跨算法对比测试
        test_cross_algorithm_comparison()
        
        print("\n" + "=" * 70)
        print("✅ 所有功能测试通过!")
        print("=" * 70)
        
        print("\n📝 测试总结:")
        print("1. ✅ 上下文感知RAG工作正常")
        print("2. ✅ 指代消解功能正常")
        print("3. ✅ 会话管理功能正常")
        print("4. ✅ MCP工具集成正常")
        print("5. ✅ 多算法学习场景正常")
        
        print("\n🎯 下一步:")
        print("1. 测试API端点: python tests/test_context_aware.py")
        print("2. 测试MCP Server: python tests/test_mcp_server.py")
        print("3. 启动RAG服务并用Postman测试")
        print("4. 在Kiro中配置MCP并测试Skills")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
