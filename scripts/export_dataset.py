"""将 MySQL 数据导出为标准 JSON 格式

这个脚本会：
1. 从数据库读取所有算法数据
2. 转换为标准 JSON 格式
3. 保存到 data/full_dataset.json
"""

import sys
import os
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db_source import fetch_algorithms


def export_full_dataset(output_file='data/full_dataset.json'):
    """导出完整数据集"""
    print("=" * 60)
    print("导出数据集到标准 JSON 格式")
    print("=" * 60)
    
    # 获取所有算法
    print("\n📥 正在从数据库读取数据...")
    algorithms = fetch_algorithms()
    print(f"✅ 成功读取 {len(algorithms)} 个算法")
    
    # 转换为标准格式
    print("\n🔄 正在转换数据格式...")
    dataset = []
    total_qa = 0
    
    for algo in algorithms:
        qa_pairs = [
            {
                'question': qa['title'],
                'answer': qa['ans']
            }
            for qa in algo['question_docs']
        ]
        
        dataset.append({
            'id': algo['algorithm_id'],
            'name': algo['algorithm_name'],
            'code': algo['code'],
            'steps': algo['step_text'],
            'analysis': algo['analysis_text'],
            'qa_pairs': qa_pairs,
            'qa_count': len(qa_pairs)
        })
        
        total_qa += len(qa_pairs)
    
    # 保存到文件
    print("\n💾 正在保存到文件...")
    os.makedirs('data', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("✅ 导出完成！")
    print("=" * 60)
    print(f"\n📊 数据统计:")
    print(f"   算法总数: {len(dataset)}")
    print(f"   问答总数: {total_qa}")
    print(f"   平均每算法问答数: {total_qa/len(dataset):.1f}")
    print(f"\n📁 保存位置: {output_file}")
    
    # 数据质量检查
    print(f"\n🔍 数据质量检查:")
    has_code = sum(1 for item in dataset if item['code'])
    has_steps = sum(1 for item in dataset if item['steps'])
    has_analysis = sum(1 for item in dataset if item['analysis'])
    has_qa = sum(1 for item in dataset if item['qa_pairs'])
    
    print(f"   有代码: {has_code}/{len(dataset)} ({has_code/len(dataset)*100:.1f}%)")
    print(f"   有步骤: {has_steps}/{len(dataset)} ({has_steps/len(dataset)*100:.1f}%)")
    print(f"   有分析: {has_analysis}/{len(dataset)} ({has_analysis/len(dataset)*100:.1f}%)")
    print(f"   有问答: {has_qa}/{len(dataset)} ({has_qa/len(dataset)*100:.1f}%)")
    
    # 显示前3个算法的示例
    print(f"\n📋 数据示例（前3个算法）:")
    for i, algo in enumerate(dataset[:3], 1):
        print(f"\n  {i}. {algo['name']} (ID: {algo['id']})")
        print(f"     - 代码长度: {len(algo['code'])} 字符")
        print(f"     - 步骤长度: {len(algo['steps'])} 字符")
        print(f"     - 分析长度: {len(algo['analysis'])} 字符")
        print(f"     - 问答数: {algo['qa_count']}")
        if algo['qa_pairs']:
            print(f"     - 第一个问答: {algo['qa_pairs'][0]['question']}")
    
    return dataset


if __name__ == '__main__':
    try:
        export_full_dataset()
        print("\n" + "=" * 60)
        print("🎉 数据导出成功！")
        print("=" * 60)
        print("\n💡 下一步:")
        print("   1. 查看导出的数据: data/full_dataset.json")
        print("   2. 运行数据集划分: python scripts/split_dataset.py")
        print("   3. 生成统计报告: python scripts/dataset_statistics.py")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的原因:")
        print("  1. 数据库连接失败 - 检查 .env 文件中的数据库配置")
        print("  2. 数据库表不存在 - 确认数据库已正确初始化")
        print("  3. 权限问题 - 确认数据库用户有读取权限")
        import traceback
        traceback.print_exc()
