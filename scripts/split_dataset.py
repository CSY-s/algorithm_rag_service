"""将数据集划分为训练/验证/测试集

按照 6:2:2 的比例划分：
- 训练集：60% (~93 个算法)
- 验证集：20% (~31 个算法)
- 测试集：20% (~31 个算法)
"""

import json
import random
import os


def split_dataset(input_file='data/full_dataset.json'):
    """按 6:2:2 划分数据集"""
    print("=" * 60)
    print("划分数据集")
    print("=" * 60)
    
    # 读取数据
    print("\n📥 正在读取数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 读取到 {len(data)} 个算法")
    
    # 设置随机种子保证可复现
    random.seed(42)
    random.shuffle(data)
    
    # 划分
    print("\n🔄 正在划分数据集...")
    n = len(data)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)
    
    train_set = data[:train_end]
    val_set = data[train_end:val_end]
    test_set = data[val_end:]
    
    # 统计问答数
    train_qa = sum(item['qa_count'] for item in train_set)
    val_qa = sum(item['qa_count'] for item in val_set)
    test_qa = sum(item['qa_count'] for item in test_set)
    
    # 保存
    print("\n💾 正在保存划分后的数据集...")
    with open('data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open('data/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    with open('data/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("✅ 划分完成！")
    print("=" * 60)
    
    print(f"\n📊 划分结果:")
    print(f"\n   训练集 (train.json):")
    print(f"      算法数: {len(train_set)} ({len(train_set)/n*100:.1f}%)")
    print(f"      问答数: {train_qa}")
    print(f"      平均每算法问答数: {train_qa/len(train_set):.1f}")
    
    print(f"\n   验证集 (val.json):")
    print(f"      算法数: {len(val_set)} ({len(val_set)/n*100:.1f}%)")
    print(f"      问答数: {val_qa}")
    print(f"      平均每算法问答数: {val_qa/len(val_set):.1f}")
    
    print(f"\n   测试集 (test.json):")
    print(f"      算法数: {len(test_set)} ({len(test_set)/n*100:.1f}%)")
    print(f"      问答数: {test_qa}")
    print(f"      平均每算法问答数: {test_qa/len(test_set):.1f}")
    
    print(f"\n   总计:")
    print(f"      算法数: {n}")
    print(f"      问答数: {train_qa + val_qa + test_qa}")
    
    # 验证划分质量
    print(f"\n🔍 划分质量检查:")
    
    # 检查是否有重叠
    train_ids = {item['id'] for item in train_set}
    val_ids = {item['id'] for item in val_set}
    test_ids = {item['id'] for item in test_set}
    
    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("   ⚠️ 警告：存在数据泄露！")
        if overlap_train_val:
            print(f"      训练集 ∩ 验证集: {len(overlap_train_val)} 个")
        if overlap_train_test:
            print(f"      训练集 ∩ 测试集: {len(overlap_train_test)} 个")
        if overlap_val_test:
            print(f"      验证集 ∩ 测试集: {len(overlap_val_test)} 个")
    else:
        print("   ✅ 无数据泄露，划分正确")
    
    # 显示每个集合的前3个算法
    print(f"\n📋 训练集示例（前3个）:")
    for i, algo in enumerate(train_set[:3], 1):
        print(f"   {i}. {algo['name']} (ID: {algo['id']}, {algo['qa_count']} 个问答)")
    
    print(f"\n📋 验证集示例（前3个）:")
    for i, algo in enumerate(val_set[:3], 1):
        print(f"   {i}. {algo['name']} (ID: {algo['id']}, {algo['qa_count']} 个问答)")
    
    print(f"\n📋 测试集示例（前3个）:")
    for i, algo in enumerate(test_set[:3], 1):
        print(f"   {i}. {algo['name']} (ID: {algo['id']}, {algo['qa_count']} 个问答)")
    
    return train_set, val_set, test_set


if __name__ == '__main__':
    try:
        split_dataset()
        
        print("\n" + "=" * 60)
        print("🎉 数据集划分成功！")
        print("=" * 60)
        print("\n📁 生成的文件:")
        print("   - data/train.json (训练集)")
        print("   - data/val.json (验证集)")
        print("   - data/test.json (测试集)")
        
        print("\n💡 下一步:")
        print("   1. 生成统计报告: python scripts/dataset_statistics.py")
        print("   2. 开始实验设计: 实现基线方法")
        print("   3. 查看数据文件: 检查划分结果")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
