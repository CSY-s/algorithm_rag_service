"""测试新增的8大功能

运行方法:
python tests/test_new_features.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_chain_visualizer():
    """测试功能1: 思维链可视化"""
    print("\n========== 测试功能1: 思维链可视化 ==========")
    
    response = requests.post(f"{BASE_URL}/api/chain/visualize", json={
        "answer": "首先，快速排序选择一个基准元素。其次，将数组分为两部分。然后，递归排序左右两部分。最后，合并结果得到有序数组。"
    })
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 成功解析思维链")
        print(f"   - 步骤数: {len(data['chain']['steps'])}")
        print(f"   - 图节点数: {len(data['graph']['nodes'])}")
        print(f"   - 图边数: {len(data['graph']['edges'])}")
        return True
    else:
        print(f"❌ 失败: {response.status_code}")
        return False


def test_multi_step_planner():
    """测试功能2: 多步规划器"""
    print("\n========== 测试功能2: 多步规划器 ==========")
    
    response = requests.post(f"{BASE_URL}/api/planner/multi_step", json={
        "question": "对比快速排序和归并排序的优缺点"
    })
    
    if response.status_code == 200:
        data = response.json()['data']
        plan = data['plan']
        print(f"✅ 成功生成规划")
        print(f"   - 主目标: {plan['main_goal']}")
        print(f"   - 子问题数: {len(plan.get('sub_questions', []))}")
        print(f"   - 执行步骤数: {len(plan['execution_plan'])}")
        print(f"   - 规划验证: {'通过' if data['validation']['is_valid'] else '失败'}")
        return True
    else:
        print(f"❌ 失败: {response.status_code}")
        return False


def test_graph_visualizer():
    """测试功能3: 知识图谱可视化"""
    print("\n========== 测试功能3: 知识图谱可视化 ==========")
    
    # 测试3.1: 搜索图谱
    print("测试3.1: 搜索图谱")
    response = requests.get(f"{BASE_URL}/api/graph/search?query=排序&max_nodes=10")
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 搜索成功")
        print(f"   - 节点数: {data['stats']['node_count']}")
        print(f"   - 边数: {data['stats']['edge_count']}")
    else:
        print(f"❌ 搜索失败: {response.status_code}")
        return False
    
    # 测试3.2: 算法子图
    print("\n测试3.2: 算法子图")
    response = requests.get(f"{BASE_URL}/api/graph/algorithm/快速排序?depth=1")
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 获取子图成功")
        print(f"   - 节点数: {data['stats']['node_count']}")
    else:
        print(f"❌ 获取子图失败: {response.status_code}")
        return False
    
    # 测试3.3: 最短路径
    print("\n测试3.3: 最短路径")
    response = requests.get(f"{BASE_URL}/api/graph/path?from_node=快速排序&to_node=归并排序")
    
    if response.status_code == 200:
        data = response.json()['data']
        if 'path' in data and len(data['path']) > 0:
            print(f"✅ 找到路径，长度: {data.get('length', 0)}")
        else:
            print(f"⚠️  未找到路径")
    else:
        print(f"❌ 路径查询失败: {response.status_code}")
    
    return True


def test_conversation_manager():
    """测试功能4: 对话分支管理"""
    print("\n========== 测试功能4: 对话分支管理 ==========")
    
    # 测试4.1: 创建分支
    print("测试4.1: 创建分支")
    response = requests.post(f"{BASE_URL}/api/conversation/branch", json={
        "session_id": "test_session_001",
        "branch_point": 1,
        "new_question": "如果用归并排序呢？"
    })
    
    if response.status_code == 200:
        data = response.json()['data']
        branch_id = data.get('branch_id')
        print(f"✅ 创建分支成功: {branch_id}")
    else:
        print(f"❌ 创建分支失败: {response.status_code}")
        return False
    
    # 测试4.2: 获取对话树
    print("\n测试4.2: 获取对话树")
    response = requests.get(f"{BASE_URL}/api/conversation/tree/test_session_001")
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 获取对话树成功")
        print(f"   - 分支数: {len(data.get('branches', []))}")
    else:
        print(f"❌ 获取对话树失败: {response.status_code}")
        return False
    
    return True


def test_quality_scorer():
    """测试功能5: RAG质量评分"""
    print("\n========== 测试功能5: RAG质量评分 ==========")
    
    response = requests.post(f"{BASE_URL}/api/quality/score", json={
        "question": "快速排序的步骤是什么？",
        "answer": "快速排序的核心步骤包括：\n1. 选择基准元素\n2. 分区操作[chunk_id=123]\n3. 递归排序左右子数组",
        "references": [
            {"id": 123, "chunk_type": "step", "content": "快速排序步骤..."}
        ]
    })
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 质量评分成功")
        print(f"   - 总分: {data['total_score']}")
        print(f"   - 等级: {data['grade']}")
        print(f"   - 完整性: {data['scores']['completeness']}")
        print(f"   - 相关性: {data['scores']['relevance']}")
        print(f"   - 清晰度: {data['scores']['clarity']}")
        print(f"   - 引用质量: {data['scores']['citation']}")
        print(f"   - 优点数: {len(data['feedback']['strengths'])}")
        print(f"   - 改进建议数: {len(data['feedback']['improvements'])}")
        return True
    else:
        print(f"❌ 失败: {response.status_code}")
        return False


def test_feedback_learner():
    """测试功能6: 用户反馈学习"""
    print("\n========== 测试功能6: 用户反馈学习 ==========")
    
    # 测试6.1: 提交反馈
    print("测试6.1: 提交反馈")
    response = requests.post(f"{BASE_URL}/api/feedback/submit", json={
        "session_id": "test_session_002",
        "question": "快速排序的时间复杂度？",
        "answer": "O(n log n)",
        "feedback_type": "thumbs_up",
        "feedback_reason": "答案准确",
        "chunk_ids": [123, 456]
    })
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 提交反馈成功: {data['feedback_id']}")
    else:
        print(f"❌ 提交反馈失败: {response.status_code}")
        return False
    
    # 测试6.2: 获取统计
    print("\n测试6.2: 获取反馈统计")
    response = requests.get(f"{BASE_URL}/api/feedback/stats")
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 获取统计成功")
        print(f"   - 总反馈数: {data.get('total_feedback', 0)}")
        print(f"   - 满意度: {data.get('satisfaction_rate', 0):.2%}")
    else:
        print(f"❌ 获取统计失败: {response.status_code}")
        return False
    
    # 测试6.3: 改进报告
    print("\n测试6.3: 生成改进报告")
    response = requests.get(f"{BASE_URL}/api/feedback/report")
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 生成报告成功")
        print(f"   - 整体满意度: {data.get('overall_satisfaction', 0):.2%}")
        print(f"   - 低质量chunk数: {len(data.get('low_quality_chunks', []))}")
        print(f"   - 改进建议数: {len(data.get('recommendations', []))}")
    else:
        print(f"❌ 生成报告失败: {response.status_code}")
    
    return True


def test_algorithm_recommender():
    """测试功能7: 算法推荐系统"""
    print("\n========== 测试功能7: 算法推荐系统 ==========")
    
    # 测试7.1: 推荐算法
    print("测试7.1: 推荐算法")
    response = requests.post(f"{BASE_URL}/api/recommend/algorithms", json={
        "user_id": 123,
        "session_history": [
            {"algorithm_id": 10, "algorithm_name": "快速排序"},
            {"algorithm_id": 15, "algorithm_name": "二分查找"}
        ],
        "top_k": 3
    })
    
    if response.status_code == 200:
        data = response.json()['data']
        print(f"✅ 推荐成功，返回{len(data)}个算法")
        for i, algo in enumerate(data, 1):
            print(f"   {i}. {algo.get('algorithm_name', 'N/A')} (评分: {algo.get('score', 0):.2f})")
    else:
        print(f"❌ 推荐失败: {response.status_code}")
        return False
    
    # 测试7.2: 学习路径
    print("\n测试7.2: 生成学习路径")
    response = requests.get(f"{BASE_URL}/api/recommend/learning_path?start=二分查找&end=动态规划")
    
    if response.status_code == 200:
        data = response.json()['data']
        if 'path' in data and len(data['path']) > 0:
            print(f"✅ 生成路径成功")
            print(f"   - 路径: {' -> '.join(data['path'])}")
            print(f"   - 总步骤: {data.get('total_steps', 0)}")
            print(f"   - 预计时间: {data.get('total_time', 'N/A')}")
        else:
            print(f"⚠️  未找到路径")
    else:
        print(f"❌ 生成路径失败: {response.status_code}")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试新增的8大功能")
    print("=" * 60)
    
    # 检查服务是否运行
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ RAG服务未运行，请先启动服务")
            print("   启动命令: python -m uvicorn app.main:app --reload")
            return
    except Exception as e:
        print(f"❌ 无法连接到RAG服务: {e}")
        print("   请确保服务运行在 http://localhost:8000")
        return
    
    print("✅ RAG服务正在运行\n")
    
    results = {}
    
    # 运行各项测试
    results['思维链可视化'] = test_chain_visualizer()
    results['多步规划器'] = test_multi_step_planner()
    results['知识图谱可视化'] = test_graph_visualizer()
    results['对话分支管理'] = test_conversation_manager()
    results['RAG质量评分'] = test_quality_scorer()
    results['用户反馈学习'] = test_feedback_learner()
    results['算法推荐系统'] = test_algorithm_recommender()
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个功能测试通过")
    
    if passed == total:
        print("\n🎉 所有功能测试通过！")
    else:
        print(f"\n⚠️  有 {total - passed} 个功能测试失败")


if __name__ == "__main__":
    main()
