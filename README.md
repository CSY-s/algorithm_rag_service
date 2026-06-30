# 算法教学智能问答系统（Algorithm RAG System）

> 基于混合检索和知识图谱增强的算法教学RAG系统  
> 已完成完整实验评测，包含155个算法、762个问答对  
> **最后更新**: 2026-06-11  
> **总功能模块**: 18个 | **API端点**: 28个 | **新增8大功能** ⭐

## 📋 项目简介

本项目是一个面向算法教学场景的检索增强生成（RAG）系统，采用**混合检索策略**（TF-IDF + 关键词 + 向量）结合**双图谱架构**（静态知识图谱 + 动态记忆图谱），实现智能化的算法问答。

### 核心特性

#### 原有核心功能（11个）

- ✅ **混合检索策略**：TF-IDF、关键词、向量检索三重融合
- ✅ **双图谱增强**：静态知识图谱（318节点+355关系）+ 动态记忆图谱
- ✅ **规划式Agent工作流** ⭐：先规划后执行，能够分解复杂问题
- ✅ **思维链推理** ⭐：可解释的推理过程
- ✅ **上下文感知**：指代消解 + 会话管理
- ✅ **工具增强模块**：复杂度提取、概念解释、上下文追踪
- ✅ **MCP Server**：模型上下文协议服务
- ✅ **Skills系统**：技能扩展框架
- ✅ **完整评测体系**：基线对比、消融实验、分层难度实验
- ✅ **LLM-as-Judge**：四维评分（准确性、完整性、清晰度、相关性）
- ✅ **数据集**：155个算法 + 762个问答对，已划分train/val/test

#### 新增8大功能（2026-06-11）⭐⭐⭐

1. ✅ **可视化思维链输出** - 前端展示推理过程，提升可解释性
2. ✅ **更强的多步规划器** - 分解复杂问题为子问题，生成执行计划
3. ✅ **知识图谱可视化** - 图形化展示算法知识关系，支持子图搜索
4. ✅ **对话分支管理** - 支持回退和多分支探索，创建学习路径
5. ✅ **RAG质量评分** - 实时评估答案质量（完整性、相关性、清晰度、引用）
6. ✅ **用户反馈学习** - 根据点赞/点踩调整系统，持续优化
7. ✅ **算法推荐系统** - 个性化学习路径，智能推荐下一步学习内容
8. ✅ **多轮对话优化** - 已在原系统中实现（会话管理+指代消解）

**总计**: 18个核心功能模块 | 28个API端点 | ~3000行新增代码

详见：[新增8大功能说明.md](./新增8大功能说明.md)

### 实验结果摘要

| 指标 | 中等问题 | 较难问题 |
|------|---------|---------|
| Judge综合分 | **4.63** (+2.2%) | 4.03 (-5.2%) |
| 概念覆盖度 | **91.64%** (+2.5%) | **96.19%** (+4.8%) |
| 解释清晰度 | **61.67%** (最高) | **83.33%** (并列第1) |

详见：[实验结果初步分析.md](./实验结果初步分析.md)

## 🚀 快速开始

### 安装依赖

```bash
# 克隆项目
cd g:\zhuomian\algorithm_rag_service

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
copy .env.example .env
# 编辑 .env 文件，填写 DEEPSEEK_API_KEY 和 MySQL 连接参数
```

### 启动服务

```bash
# 启动 API 服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 访问 Web 界面
# http://localhost:8000
```

### 首次使用

1. 打开 Web 界面：`http://localhost:8000`
2. 点击"构建知识库"按钮（约30秒）
3. 在问答框输入算法问题，如"快速排序的时间复杂度是多少？"
4. 查看答案和引用片段

## 📚 核心文档

### 功能与使用文档
| 文档 | 说明 |
|------|------|
| [新增8大功能说明.md](./新增8大功能说明.md) ⭐ | 详细的新功能说明、API文档和使用示例（5000+字） |
| [完整功能清单.md](./完整功能清单.md) | 所有18个功能模块的完整说明 |
| [新功能实现完成报告.md](./新功能实现完成报告.md) | 新功能完成情况、技术实现和测试结果 |
| [快速开始.md](./快速开始.md) | 5分钟快速上手指南 |
| [文档导航.md](./文档导航.md) | 所有文档的索引和阅读建议 |

### 集成文档
| 文档 | 说明 |
|------|------|
| [完整集成指南.md](./完整集成指南.md) | Django + Vue集成步骤 |
| [网站集成方案.md](./网站集成方案.md) | 完整的架构设计 |
| [网站集成使用示例.md](./网站集成使用示例.md) | API调用示例 |
| [MCP与Skills扩展方案.md](./MCP与Skills扩展方案.md) | MCP Server和Skills系统说明 |

### 科研文档
| 文档 | 说明 |
|------|------|
| [实验操作完整指南.md](./实验操作完整指南.md) | 所有实验的详细操作方法和结果解读 |
| [实验结果初步分析.md](./实验结果初步分析.md) | 最新实验结果的深度分析 |
| [优先级任务完成报告.md](./优先级任务完成报告.md) | 科研任务完成情况总结 |
| [科研化改进方案_v2.md](./科研化改进方案_v2.md) | 项目科研化的整体规划 |

### 技术文档
| 文档 | 说明 |
|------|------|
| [项目详细介绍文档.md](./项目详细介绍文档.md) | 完整的技术实现细节 |
| [数据库操作说明.md](./数据库操作说明.md) | 数据库安全性说明 |
| [功能检查列表.md](./功能检查列表.md) | 所有功能的检查清单 |

## 🧪 实验与评测

### 已完成的实验

1. **基线对比实验**：对比5种方法（BM25、纯LLM、标准RAG、TF-IDF+LLM、本系统）
2. **消融实验**：验证各模块贡献（工具8.47%、图谱6.38%、记忆5.55%、混合检索3.80%）
3. **分层难度实验**：按问题难度（简单20条、中等15条、较难10条）分层评测

### 运行实验

```bash
# 基线对比实验
.venv\Scripts\python.exe scripts\run_baseline_comparison.py

# 消融实验
.venv\Scripts\python.exe scripts\run_ablation_study.py

# 分层实验（含LLM-Judge）
.venv\Scripts\python.exe scripts\run_layered_experiment.py

# 查看结果
type data\results\layered_experiment_*.md
```

详见：[实验操作完整指南.md](./实验操作完整指南.md)

## 🎯 主要API接口

### 核心接口（11个原有接口）

- `GET /health` - 健康检查
- `GET /stats` - 系统统计
- `POST /build` - 构建知识库
  ```json
  {"with_synth": true, "limit": null}
  ```
- `POST /ask` - 智能问答（完整功能）
  ```json
  {
    "question": "快速排序的时间复杂度是多少？",
    "top_k": 5,
    "algorithm_id": null,
    "enable_tools": true,
    "enable_memory": true,
    "enable_planning": true
  }
  ```
- `POST /search` - 检索调试
- `POST /evaluate` - 评测对比
- `POST /experiment` - 批量实验
- `POST /api/rag/context_ask` - 上下文问答
- `GET /api/rag/session/{id}` - 会话历史
- `POST /memory_graph` - 记忆子图
- `POST /synthesize` - 数据合成

### 新增接口（17个新接口）⭐

#### 思维链可视化（1个）
- `POST /api/chain/visualize` - 可视化思维链步骤

#### 高级规划（1个）
- `POST /api/planner/multi_step` - 多步规划

#### 知识图谱可视化（4个）
- `GET /api/graph/search` - 搜索图谱
- `GET /api/graph/algorithm/{name}` - 算法子图
- `GET /api/graph/full` - 完整图谱
- `GET /api/graph/path` - 最短路径

#### 对话分支管理（3个）
- `POST /api/conversation/branch` - 创建分支
- `POST /api/conversation/rollback` - 回退对话
- `GET /api/conversation/tree/{id}` - 对话树

#### RAG质量评分（2个）
- `POST /api/quality/score` - 评分答案
- `POST /api/quality/compare` - 对比答案

#### 用户反馈学习（4个）
- `POST /api/feedback/submit` - 提交反馈
- `GET /api/feedback/stats` - 反馈统计
- `GET /api/feedback/low_quality` - 低质量chunk
- `GET /api/feedback/report` - 改进报告

#### 算法推荐（2个）
- `POST /api/recommend/algorithms` - 推荐算法
- `GET /api/recommend/learning_path` - 学习路径

**总计**: 28个API端点

完整API文档：启动服务后访问 `http://localhost:8000/docs`  
详细说明：查看 [新增8大功能说明.md](./新增8大功能说明.md)

## 🏗️ 项目架构

### 目录结构

```
algorithm_rag_service/
├─ app/                      # 核心代码
│  ├─ main.py               # FastAPI 入口
│  ├─ rag.py                # RAG 核心逻辑
│  ├─ hybrid_retrieval.py   # 混合检索实现
│  ├─ baselines.py          # 基线方法
│  ├─ metrics.py            # ROUGE/BLEU 指标
│  ├─ advanced_metrics.py   # BERTScore/概念覆盖度等
│  ├─ llm_judge.py          # LLM-as-Judge 评估
│  ├─ memory_graph.py       # 动态记忆图谱
│  ├─ tools.py              # 工具模块
│  └─ static/index.html     # Web 界面
├─ scripts/                  # 实验脚本
│  ├─ export_dataset.py     # 数据导出
│  ├─ split_dataset.py      # 数据集划分
│  ├─ run_baseline_comparison.py    # 基线对比实验
│  ├─ run_ablation_study.py         # 消融实验
│  └─ run_layered_experiment.py     # 分层难度实验
├─ data/                     # 数据目录
│  ├─ train.json            # 训练集（60%）
│  ├─ val.json              # 验证集（20%）
│  ├─ test.json             # 测试集（20%）
│  └─ results/              # 实验结果
├─ docs/                     # 参考文档
└─ *.md                      # 核心文档
```

### 系统架构

```
用户问题
    ↓
【问题扩展】(可选)
    ↓
【混合检索】TF-IDF + 关键词 + 向量
    ↓
【知识图谱增强】静态图谱 + 动态记忆
    ↓
【工具增强】复杂度提取 + 概念解释
    ↓
【LLM 生成】DeepSeek-Chat
    ↓
【答案 + 引用】
```

## 📊 数据集

| 项目 | 数量 | 说明 |
|------|------|------|
| 算法总数 | 155 | 覆盖排序、查找、图、树、动态规划等 |
| 问答对 | 762 | 真实问答数据 |
| 知识图谱节点 | 318 | 算法、概念、数据结构 |
| 知识图谱关系 | 355 | 概念关联、应用场景 |
| 训练集 | 93算法 | 60% |
| 验证集 | 31算法 | 20% |
| 测试集 | 31算法 | 20% |

## 🔬 技术栈

- **后端**: Python 3.8+, FastAPI
- **数据库**: MySQL
- **LLM**: DeepSeek-Chat
- **检索**: scikit-learn (TF-IDF), jieba (分词)
- **评测**: rouge-chinese, nltk (BLEU), bert-score
- **前端**: 原生 HTML/CSS/JavaScript

## 📖 使用示例

### 1. 构建知识库

```python
import requests

response = requests.post("http://localhost:8000/build", json={
    "with_synth": True,
    "limit": None
})
print(response.json())
```

### 2. 问答示例

```python
response = requests.post("http://localhost:8000/ask", json={
    "question": "快速排序的平均时间复杂度是多少？",
    "top_k": 5,
    "enable_tools": True,
    "enable_memory": True
})

result = response.json()
print(f"答案: {result['answer']}")
print(f"引用: {result['retrieved_chunks']}")
```

### 3. 运行评测

```python
response = requests.post("http://localhost:8000/evaluate", json={
    "limit": 30,
    "top_k": 5
})

metrics = response.json()
print(f"ROUGE-L: {metrics['rag_plus']['rouge']['rouge_l']}")
```

## 🎓 科研成果

### 核心贡献

1. 提出三重混合检索策略，消融实验证明贡献3.80%
2. 设计双图谱架构（静态+动态），贡献6.38%
3. 构建高质量算法教学数据集（155算法，762问答）
4. 在中等问题上LLM-Judge评分提升2.2%
5. 在概念覆盖度上提升4.8%（达到96.19%）

### 适合投稿

- **中文期刊**: 《中文信息学报》（T2）⭐⭐⭐⭐⭐
- **教育AI会议**: AIED/EDM Workshop ⭐⭐⭐⭐
- **应用会议**: CCIR（中国信息检索会议）⭐⭐⭐

详见：[优先级任务完成报告.md](./优先级任务完成报告.md)

## 📝 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@software{algorithm_rag_2026,
  title={Algorithm Teaching Intelligent QA System with Hybrid Retrieval and Knowledge Graph Enhancement},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/algorithm_rag_service}
}
```

## 📄 许可证

[MIT License](LICENSE)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

- 项目主页: [GitHub](https://github.com/yourusername/algorithm_rag_service)
- 问题反馈: [Issues](https://github.com/yourusername/algorithm_rag_service/issues)

---

**最后更新**: 2026-06-11  
**版本**: v3.0  
**状态**: 实验已完成，新增8大功能，可投入使用

</content>
</file>
<file name="文档导航.md" language="markdown" >
<content>
# 📚 项目文档导航

> **最后更新**: 2026-06-11  
> **文档总数**: 15个核心文档 + 3个参考文档
