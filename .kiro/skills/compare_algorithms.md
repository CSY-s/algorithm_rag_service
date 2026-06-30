---
name: compare_algorithms
description: 对比两个算法的异同
parameters:
  - name: algorithm1
    type: string
    required: true
  - name: algorithm2
    type: string
    required: true
  - name: focus_aspects
    type: array
    required: false
    description: 重点对比的方面
---

# 对比：{{algorithm1}} vs {{algorithm2}}

请从以下维度对比这两个算法：

## 对比表格

| 维度 | {{algorithm1}} | {{algorithm2}} | 说明 |
|------|----------------|----------------|------|
| 时间复杂度 | | | |
| 空间复杂度 | | | |
| 稳定性 | | | |
| 实现难度 | | | |

## 详细分析

### 1. 性能对比
在什么情况下{{algorithm1}}更快？
在什么情况下{{algorithm2}}更快？

### 2. 使用场景
各自适合什么场景？

### 3. 实际应用
在工业界/竞赛中的使用情况。

---

**使用的工具**:
- `compare_algorithms("{{algorithm1}}", "{{algorithm2}}")`
- `get_algorithm_code("{{algorithm1}}")`
- `get_algorithm_code("{{algorithm2}}")`
