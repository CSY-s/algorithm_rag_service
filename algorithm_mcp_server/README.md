# Algorithm Teaching MCP Server

算法教学MCP服务器，提供155个算法的检索、解释和对比功能。

## 快速开始

### 安装依赖

```bash
pip install fastmcp mcp
```

### 启动服务器

```bash
python mcp_server.py
```

## 提供的工具

1. **search_algorithm** - 搜索算法知识库
2. **get_algorithm_code** - 获取算法代码
3. **explain_algorithm** - 解释算法原理
4. **compare_algorithms** - 对比两个算法
5. **list_algorithms** - 列出所有算法

## 提供的资源

1. **algorithms://list** - 算法列表
2. **algorithms://{name}** - 单个算法详情

## 提供的提示词

1. **explain_algorithm_prompt** - 解释算法的提示词
2. **compare_algorithms_prompt** - 对比算法的提示词

## 在Kiro中使用

创建 `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "algorithm-teaching": {
      "command": "python",
      "args": ["G:/zhuomian/algorithm_rag_service/algorithm_mcp_server/mcp_server.py"],
      "disabled": false,
      "autoApprove": [
        "search_algorithm",
        "list_algorithms"
      ]
    }
  }
}
```

## 示例

```python
# 搜索排序算法
search_algorithm("排序", top_k=5)

# 获取快速排序代码
get_algorithm_code("快速排序")

# 解释归并排序
explain_algorithm("归并排序", detail_level="detailed")

# 对比快排和归并
compare_algorithms("快速排序", "归并排序")
```
