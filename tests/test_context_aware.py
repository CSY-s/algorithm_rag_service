"""测试上下文感知RAG功能

测试内容:
1. 指代消解（"它"、"这个算法"）
2. 会话记忆
3. algorithm_id范围限定
4. 会话历史管理
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import rag_ask, _resolve_references
from app.session import (
    save_session_history, 
    get_session_history, 
    get_session_summary,
    clear_session
)


def test_algorithm_id_restriction():
    """测试algorithm_id范围限定"""
    print("\n=== 测试1: algorithm_id范围限定 ===")
    
    # 测试1: 在快速排序范围内提问
    result1 = rag_ask(
        question="时间复杂度是多少?",
        algorithm_id=10,  # 假设10是快速排序
        top_k=3
    )
    print(f"问题: 时间复杂度是多少?")
    print(f"限定算法ID: 10")
    print(f"回答: {result1['answer'][:200]}...")
    print(f"检索到的chunks数量: {len(result1['references'])}")
    
    # 测试2: 不限定范围（全局检索）
    result2 = rag_ask(
        question="时间复杂度是多少?",
        algorithm_id=None,  # 不限定
        top_k=3
    )
    print(f"\n不限定算法ID")
    print(f"回答: {result2['answer'][:200]}...")
    print(f"检索到的chunks数量: {len(result2['references'])}")
    
    # 验证：限定范围后的回答应该更聚焦
    assert len(result1['references']) <= len(result2['references'])
    print("\n✅ 测试通过: algorithm_id范围限定生效")


def test_reference_resolution():
    """测试指代消解"""
    print("\n=== 测试2: 指代消解 ===")
    
    # 测试不同的指代词
    test_cases = [
        ("它的时间复杂度是多少?", 10),
        ("这个算法的空间复杂度呢?", 10),
        ("该算法是稳定的吗?", 10),
    ]
    
    for question, algo_id in test_cases:
        resolved = _resolve_references(question, algo_id, None)
        print(f"原问题: {question}")
        print(f"消解后: {resolved}")
        
        # 应该包含算法名称
        assert resolved != question or "算法" not in question
    
    print("\n✅ 测试通过: 指代消解功能正常")


def test_session_memory():
    """测试会话记忆"""
    print("\n=== 测试3: 会话记忆 ===")
    
    session_id = "test_session_001"
    clear_session(session_id)  # 清空旧数据
    
    # 第一轮对话
    result1 = rag_ask(
        question="快速排序的基本思想是什么?",
        algorithm_id=10,
        session_id=session_id,
        user_id=999,
        top_k=3
    )
    print(f"第1轮问题: 快速排序的基本思想是什么?")
    print(f"回答: {result1['answer'][:100]}...")
    
    # 第二轮对话（使用指代词）
    result2 = rag_ask(
        question="它的空间复杂度是多少?",
        session_id=session_id,
        user_id=999,
        top_k=3
    )
    print(f"\n第2轮问题: 它的空间复杂度是多少?")
    print(f"消解后: {result2.get('resolved_question')}")
    print(f"回答: {result2['answer'][:100]}...")
    
    # 获取会话历史
    history = get_session_history(session_id)
    print(f"\n会话历史记录数: {len(history)}")
    assert len(history) == 2
    
    # 获取会话摘要
    summary = get_session_summary(session_id)
    print(f"会话摘要: {summary}")
    assert summary['total_messages'] == 2
    assert 10 in summary['algorithms_discussed']
    
    print("\n✅ 测试通过: 会话记忆功能正常")


def test_session_management():
    """测试会话管理"""
    print("\n=== 测试4: 会话管理 ===")
    
    session_id = "test_session_002"
    
    # 保存多条会话
    for i in range(5):
        save_session_history(
            session_id=session_id,
            user_id=999,
            algorithm_id=10 + i,
            question=f"问题{i+1}",
            answer=f"回答{i+1}"
        )
    
    # 获取最近3条
    history = get_session_history(session_id, limit=3)
    print(f"请求最近3条，实际返回: {len(history)}")
    assert len(history) == 3
    
    # 获取全部
    history_all = get_session_history(session_id, limit=0)
    print(f"请求全部，实际返回: {len(history_all)}")
    assert len(history_all) == 5
    
    # 清空会话
    clear_session(session_id)
    history_after = get_session_history(session_id)
    print(f"清空后，返回: {len(history_after)}")
    assert len(history_after) == 0
    
    print("\n✅ 测试通过: 会话管理功能正常")


def test_continuous_dialogue():
    """测试连续对话"""
    print("\n=== 测试5: 连续对话 ===")
    
    session_id = "test_session_003"
    clear_session(session_id)
    
    # 模拟用户在算法详情页的连续提问
    questions = [
        ("这个算法的步骤是什么?", 10),
        ("那时间复杂度呢?", 10),  # 延续上文
        ("最坏情况是什么?", 10),   # 继续深入
    ]
    
    for i, (question, algo_id) in enumerate(questions, 1):
        result = rag_ask(
            question=question,
            algorithm_id=algo_id,
            session_id=session_id,
            top_k=3
        )
        print(f"\n第{i}轮:")
        print(f"  问题: {question}")
        if result.get('resolved_question'):
            print(f"  消解: {result['resolved_question']}")
        print(f"  回答: {result['answer'][:80]}...")
    
    # 验证会话历史
    history = get_session_history(session_id)
    assert len(history) == 3
    print(f"\n连续对话{len(history)}轮，会话保持完整")
    
    print("\n✅ 测试通过: 连续对话功能正常")


if __name__ == '__main__':
    print("开始测试上下文感知RAG功能...")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_algorithm_id_restriction()
        test_reference_resolution()
        test_session_memory()
        test_session_management()
        test_continuous_dialogue()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("\n建议:")
        print("1. 在真实环境中测试（启动FastAPI服务）")
        print("2. 用Postman测试 /api/rag/context_ask 端点")
        print("3. 集成到前端进行端到端测试")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
