---
name: explain_algorithm
description: 详细解释一个算法的原理、步骤和复杂度
parameters:
  - name: algorithm_name
    type: string
    required: true
    description: 要解释的算法名称
  - name: detail_level
    type: string
    required: false
    default: medium
    options: [simple, medium, detailed]
    description: 详细程度
---

# 解释算法：{{algorithm_name}}

请按以下结构详细解释{{algorithm_name}}算法：

## 1. 基本思想
用一句话概括算法的核心思想。

## 2. 算法步骤
详细列出算法的执行步骤（{{detail_level}}级别）。

## 3. 复杂度分析
- 时间复杂度：最好/平均/最坏
- 空间复杂度
- 复杂度的推导过程

## 4. 典型应用
说明这个算法在实际中的应用场景。

## 5. 优缺点
- 优点
- 缺点
- 适用条件

## 6. 代码示例
提供Python实现（带注释）。

---

**使用的工具**:
- `search_algorithm("{{algorithm_name}}")`
- `get_algorithm_code("{{algorithm_name}}")`
- `explain_algorithm("{{algorithm_name}}", detail_level="{{detail_level}}")`

**参考资源**:
- `algorithms://{{algorithm_name}}`
