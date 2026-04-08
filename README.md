# 独立项目：数据库算法 RAG + 知识合成（DeepSeek）

## 功能
- 从 MySQL 读取 `algorithm_algorithm`（算法代码）与 `algorithm_question`（步骤/分析/题库问答）
- 构建 RAG 知识库到 `algorithm_rag_chunk`
- 用 DeepSeek 合成问答（可选，数据合成）
- 基于标题-资料-大纲-多视角对话的数据合成流水线，生成高质量问答对
- 检索增强问答（支持问题扩展、工具增强、会话记忆）
- 结构化记忆子图检索，将模型记忆组织为 YAML 形式参与推理
- 评测对比（`baseline_extractive` / `rag_basic` / `rag_plus`）
- 实验对比（不同 `retrieval_mode`、不同 `top_k` 自动跑批并输出报告）
- 检索调试与系统统计（`/search`、`/stats`）

## 启动
```bash
cd g:\zhuomian\algorithm_rag_service
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# 填写 DEEPSEEK_API_KEY 和 MySQL 连接参数
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

## API
- `GET /health`
- `GET /stats`
- `POST /build`
  - body: `{"with_synth": true, "limit": 100}`
- `POST /memory_graph`
  - body: `{"question":"KMP 的 next 数组如何推导？","algorithm_id":10,"top_k":6}`
  - 返回：结构化记忆三元组候选、邻居子图与 YAML 表达
- `POST /synthesize`
  - body: `{"limit":5,"algorithm_id":10}`
  - 返回：多视角对话式数据合成结果，并写回知识库
- `POST /search`
  - body: `{"question":"KMP的next怎么求？","top_k":5,"algorithm_id":10,"retrieval_mode":"hybrid","use_expansion":true,"include_memory":true}`
- `POST /ask`
  - body: `{"question":"KMP的next怎么求？","top_k":5,"algorithm_id":10,"retrieval_mode":"hybrid","current_state":"用户正在学习KMP章节","enable_tools":true,"enable_memory":true,"enable_planning":true,"enable_mcp":true}`
- `POST /evaluate`
  - body: `{"limit":12,"top_k":5,"algorithm_id":10,"retrieval_mode":"hybrid"}`
  - 返回：`baseline_extractive`、`rag_basic`、`rag_plus` 三种方案的 `ROUGE-1/2/L` 与 `BLEU-4` 平均分
- `POST /experiment`
  - body: `{"limit":8,"algorithm_id":10,"top_k_values":[3,5,8],"retrieval_modes":["tfidf","keyword","vector","hybrid"]}`
  - 返回：多配置实验结果、最佳配置、以及 `data/results/` 下的 JSON/Markdown 报告路径

## 项目结构图（文字版）
```text
algorithm_rag_service/
├─ app/
│  ├─ main.py            # FastAPI 入口：/build /ask /evaluate
│  ├─ rag.py             # 构建、检索、问答、记忆主流程
│  ├─ db_source.py       # MySQL 数据读取（算法、题库、知识图谱、评测样本）
│  ├─ store.py           # algorithm_rag_chunk 的读写与来源过滤
│  ├─ deepseek.py        # DeepSeek 调用 + 问题扩展 + 三元组抽取 + 合成 QA
│  ├─ hybrid_retrieval.py# 混合检索：TF-IDF / keyword / vector / hybrid
│  ├─ tools.py           # 工具模块（复杂度提取、知识图谱检索）
│  ├─ metrics.py         # ROUGE-1/2/L、BLEU-4 指标实现
│  ├─ evaluate.py        # 多方案评测执行与汇总
│  ├─ experiment.py      # 实验对比、检索调试、系统统计
│  ├─ synthesis_pipeline.py # 标题->资料->大纲->多视角对话->最终答案
│  ├─ memory_graph.py    # 结构化记忆子图检索与YAML组织
│  └─ static/index.html  # 前端控制台（构建/提问/评测）
├─ requirements.txt
└─ README.md
```

## 数据流图（Build / Ask / Evaluate）
```text
[MySQL:tutor]
  ├─ algorithm_algorithm
  ├─ algorithm_question
  ├─ knowledge_node / knowledge_relation
  └─ algorithm_rag_chunk

1) Build:
algorithm_algorithm + algorithm_question
    -> 分块(step/analysis/code/question_doc/qa)
    -> 写入 algorithm_rag_chunk

1.5) Synthesize:
教材标题/算法标题
    -> 检索相关资料
    -> 生成初始大纲
    -> 生成多视角提问者-专家对话
    -> 细化大纲
    -> 生成最终答案
    -> 写回知识库

2) Ask:
用户问题
    -> 问题扩展(可选, DeepSeek)
    -> 检索结构化记忆子图(YAML)
    -> 检索(top_k, TF-IDF / keyword / vector / hybrid)
    -> 工具模块(可选: 复杂度提取 + 知识图谱检索)
    -> 与当前状态 S 共同组成混合知识
    -> DeepSeek 链式推理并生成答案
    -> 三元组抽取并写入 memory chunk(可选)

3) Evaluate:
algorithm_question 生成评测样本(question/reference_answer)
    -> baseline_extractive / rag_basic / rag_plus
    -> 计算 ROUGE-1/2/L + BLEU-4
    -> 返回平均分和 case 明细

4) Experiment:
不同 retrieval_mode / top_k 组合
    -> 批量调用 evaluate
    -> 汇总配置表现
    -> 输出 JSON + Markdown 报告到 data/results
```

## 三种评测方法定义
- `baseline_extractive`：关闭增强能力，仅检索后抽取片段拼接为答案（不走生成）
- `rag_basic`：基础 RAG（检索 + LLM 生成），不启用扩展检索与工具增强
- `rag_plus`：增强 RAG（扩展检索 + 工具模块 + LLM 生成）

## 检索模式
- `tfidf`：原有字符级 TF-IDF 检索
- `keyword`：中文分词 + 标题命中 + 关键词覆盖
- `vector`：TF-IDF 稠密降维后的向量式检索
- `hybrid`：综合 `tfidf + keyword + vector`，默认推荐

## 答辩可讲的亮点
- 数据闭环：课程算法数据 -> 知识库 -> 问答 -> 记忆更新 -> 再检索
- 工程可落地：核心能力模块化，接口清晰，支持独立调试
- 可量化验证：不仅展示效果，还提供多方案自动评测对比

## 快速演示脚本（3分钟）
1. 打开页面，先点击“构建知识库”
2. 提问一个算法问题，展示答案中的引用片段
3. 勾选/取消“工具模块、记忆模块”再提一次，说明增强效果
4. 点击“运行评测”，展示三种方法分数对比（通常 `rag_plus` 更好）

## Web 界面
- 启动服务后打开：`http://127.0.0.1:8010/`
- 页面支持：
  - 一键调用 `/build` 构建知识库
  - 一键调用 `/stats` 查看系统统计
  - 一键调用 `/memory_graph` 查看结构化记忆
  - 一键调用 `/search` 只看检索结果
  - 输入问题调用 `/ask` 并展示答案和引用片段
  - 一键调用 `/evaluate`，表格展示 baseline / rag_basic / rag_plus 指标对比
  - 一键调用 `/experiment`，查看不同检索模式和 Top K 的最佳配置
  - 一键调用 `/synthesize`，生成多视角合成问答
