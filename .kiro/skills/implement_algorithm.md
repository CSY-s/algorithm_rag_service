---
name: implement_algorithm
description: 从零实现一个算法
parameters:
  - name: algorithm_name
    type: string
    required: true
  - name: language
    type: string
    required: false
    default: python
    options: [python, java, cpp, javascript]
  - name: with_tests
    type: boolean
    required: false
    default: true
---

# 实现：{{algorithm_name}}

## 任务
从零实现{{algorithm_name}}算法（{{language}}语言）。

## 要求

### 1. 代码结构
```{{language}}
// 主函数签名
function {{algorithm_name}}(input) {
    // TODO: 实现
}
```

### 2. 实现步骤
- [ ] 定义函数签名
- [ ] 实现核心逻辑
- [ ] 处理边界情况
- [ ] 添加注释

### 3. 测试用例（如果with_tests=true）
```{{language}}
// 测试用例1
input: ...
expected: ...

// 测试用例2
input: ...
expected: ...
```

### 4. 复杂度标注
在代码注释中标注：
- 时间复杂度：O(?)
- 空间复杂度：O(?)

---

**参考**:
先调用`get_algorithm_code("{{algorithm_name}}")`获取参考实现，
然后根据{{language}}语言的特点重新实现。

**验证**:
如果with_tests=true，使用测试用例验证实现的正确性。
