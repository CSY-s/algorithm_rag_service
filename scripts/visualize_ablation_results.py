"""生成消融实验结果的可视化图表。

包括：
1. 模块贡献柱状图
2. 多指标对比雷达图
3. 性能下降对比图
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 读取实验结果
with open('data/results/ablation_study_20260526_091655.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
contributions = data['contributions']

# 提取数据
configs = ['完整系统', '去掉混合检索', '去掉知识图谱', '去掉记忆机制', '去掉工具增强']
config_keys = ['full', 'no_hybrid', 'no_kg', 'no_memory', 'no_tools']

rouge_l = [results[k]['rouge']['rouge_l'] for k in config_keys]
bleu_4 = [results[k]['bleu_4'] for k in config_keys]
f1_score = [results[k]['advanced_metrics']['f1_score'] for k in config_keys]
concept_coverage = [results[k]['advanced_metrics']['concept_coverage'] for k in config_keys]

# ============================================================================
# 图1：模块贡献柱状图
# ============================================================================
plt.figure(figsize=(10, 6))

modules = ['工具增强', '知识图谱', '记忆机制', '混合检索']
contrib_values = [
    contributions['tools'],
    contributions['knowledge_graph'],
    contributions['memory'],
    contributions['hybrid_retrieval']
]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
bars = plt.bar(modules, contrib_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, value in zip(bars, contrib_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.2f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('贡献度 (%)', fontsize=14, fontweight='bold')
plt.title('消融实验：各模块对系统性能的贡献', fontsize=16, fontweight='bold', pad=20)
plt.ylim(0, max(contrib_values) * 1.2)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# 添加说明
plt.text(0.5, 0.95, '贡献度 = (完整系统ROUGE-L - 去掉模块后ROUGE-L) / 完整系统ROUGE-L × 100%',
         transform=plt.gca().transAxes, ha='center', va='top',
         fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('data/results/ablation_contribution.png', dpi=300, bbox_inches='tight')
print("✅ 图1已保存: data/results/ablation_contribution.png")
plt.close()

# ============================================================================
# 图2：ROUGE-L性能对比柱状图
# ============================================================================
plt.figure(figsize=(12, 6))

x = np.arange(len(configs))
width = 0.6

bars = plt.bar(x, rouge_l, width, color=['#2ecc71', '#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6'],
               alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, (bar, value) in enumerate(zip(bars, rouge_l)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加下降百分比（除了完整系统）
    if i > 0:
        decrease = (rouge_l[0] - value) / rouge_l[0] * 100
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'-{decrease:.1f}%',
                 ha='center', va='center', fontsize=10, color='white', fontweight='bold')

plt.ylabel('ROUGE-L Score', fontsize=14, fontweight='bold')
plt.title('消融实验：ROUGE-L性能对比', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, configs, fontsize=11, rotation=15, ha='right')
plt.ylim(0, max(rouge_l) * 1.15)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# 添加基线
plt.axhline(y=rouge_l[0], color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5, label='完整系统基线')
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('data/results/ablation_rouge_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 图2已保存: data/results/ablation_rouge_comparison.png")
plt.close()

# ============================================================================
# 图3：多指标雷达图
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 准备数据（归一化到0-1）
categories = ['ROUGE-L', 'BLEU-4', 'F1-Score', '概念覆盖度']
N = len(categories)

# 归一化函数
def normalize(values):
    max_val = max(values)
    return [v / max_val for v in values]

# 为每个配置创建数据
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 绘制每个配置
colors_radar = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
line_styles = ['-', '--', '-.', ':', '-']

for i, (config, key) in enumerate(zip(configs, config_keys)):
    values = [
        results[key]['rouge']['rouge_l'],
        results[key]['bleu_4'],
        results[key]['advanced_metrics']['f1_score'],
        results[key]['advanced_metrics']['concept_coverage']
    ]
    
    # 归一化
    values_norm = normalize(values)
    values_norm += values_norm[:1]  # 闭合
    
    ax.plot(angles, values_norm, 'o-', linewidth=2, label=config,
            color=colors_radar[i], linestyle=line_styles[i], markersize=6)
    ax.fill(angles, values_norm, alpha=0.1, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

plt.title('消融实验：多指标性能对比（归一化）', fontsize=16, fontweight='bold', pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.tight_layout()
plt.savefig('data/results/ablation_radar.png', dpi=300, bbox_inches='tight')
print("✅ 图3已保存: data/results/ablation_radar.png")
plt.close()

# ============================================================================
# 图4：多指标对比（分组柱状图）
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(configs))
width = 0.2

# 归一化数据以便对比
rouge_l_norm = [v / max(rouge_l) for v in rouge_l]
bleu_4_norm = [v / max(bleu_4) for v in bleu_4]
f1_norm = [v / max(f1_score) for v in f1_score]
concept_norm = [v / max(concept_coverage) for v in concept_coverage]

bars1 = ax.bar(x - 1.5*width, rouge_l_norm, width, label='ROUGE-L', color='#3498db', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, bleu_4_norm, width, label='BLEU-4', color='#e74c3c', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, f1_norm, width, label='F1-Score', color='#2ecc71', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, concept_norm, width, label='概念覆盖度', color='#f39c12', alpha=0.8)

ax.set_ylabel('归一化分数', fontsize=14, fontweight='bold')
ax.set_title('消融实验：多指标性能对比（归一化）', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11, rotation=15, ha='right')
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('data/results/ablation_multi_metrics.png', dpi=300, bbox_inches='tight')
print("✅ 图4已保存: data/results/ablation_multi_metrics.png")
plt.close()

# ============================================================================
# 图5：性能下降热力图
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# 准备数据
metrics = ['ROUGE-L', 'BLEU-4', 'F1-Score']
modules_heatmap = ['去掉混合检索', '去掉知识图谱', '去掉记忆机制', '去掉工具增强']

# 计算性能下降百分比
decrease_data = []
for key in ['no_hybrid', 'no_kg', 'no_memory', 'no_tools']:
    row = [
        (results['full']['rouge']['rouge_l'] - results[key]['rouge']['rouge_l']) / results['full']['rouge']['rouge_l'] * 100,
        (results['full']['bleu_4'] - results[key]['bleu_4']) / results['full']['bleu_4'] * 100,
        (results['full']['advanced_metrics']['f1_score'] - results[key]['advanced_metrics']['f1_score']) / results['full']['advanced_metrics']['f1_score'] * 100
    ]
    decrease_data.append(row)

decrease_data = np.array(decrease_data)

im = ax.imshow(decrease_data, cmap='Reds', aspect='auto', vmin=0, vmax=20)

# 设置刻度
ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(modules_heatmap)))
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.set_yticklabels(modules_heatmap, fontsize=12)

# 添加数值标签
for i in range(len(modules_heatmap)):
    for j in range(len(metrics)):
        text = ax.text(j, i, f'{decrease_data[i, j]:.1f}%',
                      ha="center", va="center", color="black", fontsize=11, fontweight='bold')

ax.set_title('消融实验：性能下降热力图', fontsize=16, fontweight='bold', pad=20)
fig.colorbar(im, ax=ax, label='性能下降 (%)', shrink=0.8)

plt.tight_layout()
plt.savefig('data/results/ablation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ 图5已保存: data/results/ablation_heatmap.png")
plt.close()

print("\n" + "="*80)
print("所有图表生成完成！")
print("="*80)
print("\n生成的图表：")
print("1. ablation_contribution.png - 模块贡献柱状图")
print("2. ablation_rouge_comparison.png - ROUGE-L性能对比")
print("3. ablation_radar.png - 多指标雷达图")
print("4. ablation_multi_metrics.png - 多指标分组柱状图")
print("5. ablation_heatmap.png - 性能下降热力图")
print("\n这些图表可以直接用于论文和报告！")
