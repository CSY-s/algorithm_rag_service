"""FastAPI 接口入口。

这个文件主要负责定义请求参数，并把 HTTP 请求转发到具体业务模块。"""


import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from .evaluate import evaluate_rag
from .experiment import get_system_stats, run_experiment_suite, search_only
from .memory_graph import select_memory_subgraph
from .rag import build_knowledge, rag_ask
from .hybrid_retrieval import RETRIEVAL_MODES
from .agent_workflow import answer_with_plan
from .synthesis_pipeline import synthesize_dialogue_qa
from .session import get_session_history, get_session_summary, clear_session

app = FastAPI(title='Algorithm RAG Service', version='1.1.0')

# 配置CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",  # Vue开发服务器
        "http://localhost:3000",  # React开发服务器
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
        # 生产环境域名请在此添加
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(BASE_DIR, 'static', 'index.html')


class BuildReq(BaseModel):
    """知识库构建接口的请求体。"""
    with_synth: bool = True
    limit: int | None = None


class SynthesisReq(BaseModel):
    """数据合成接口的请求体。"""
    limit: int = 5
    algorithm_id: int | None = None


class AskReq(BaseModel):
    """问答接口的请求体，集中管理问答流程里的各种开关。"""
    question: str
    top_k: int = 5
    algorithm_id: int | None = None
    retrieval_mode: str = 'hybrid'
    current_state: str = ''
    enable_tools: bool = True
    enable_memory: bool = True
    enable_planning: bool = True
    enable_mcp: bool = True

    @field_validator('retrieval_mode')
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        """统一校验检索模式是否合法。"""
        mode = (value or 'hybrid').lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f'retrieval_mode必须是: {", ".join(sorted(RETRIEVAL_MODES))}')
        return mode


class EvalReq(BaseModel):
    """评测接口的请求体。"""
    limit: int = 12
    top_k: int = 5
    algorithm_id: int | None = None
    retrieval_mode: str = 'hybrid'

    @field_validator('retrieval_mode')
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        """统一校验检索模式是否合法。"""
        mode = (value or 'hybrid').lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f'retrieval_mode必须是: {", ".join(sorted(RETRIEVAL_MODES))}')
        return mode


class SearchReq(BaseModel):
    """仅检索调试接口的请求体。"""
    question: str
    top_k: int = 5
    algorithm_id: int | None = None
    retrieval_mode: str = 'hybrid'
    use_expansion: bool = True
    include_memory: bool = True

    @field_validator('retrieval_mode')
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        """统一校验检索模式是否合法。"""
        mode = (value or 'hybrid').lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f'retrieval_mode必须是: {", ".join(sorted(RETRIEVAL_MODES))}')
        return mode


class MemoryReq(BaseModel):
    """结构化记忆子图查询接口的请求体。"""
    question: str
    algorithm_id: int | None = None
    top_k: int = 6


class ExperimentReq(BaseModel):
    """实验批跑接口的请求体。"""
    limit: int = 8
    algorithm_id: int | None = None
    top_k_values: list[int] | None = None
    retrieval_modes: list[str] | None = None

    @field_validator('retrieval_modes')
    @classmethod
    def validate_retrieval_modes(cls, value: list[str] | None) -> list[str] | None:
        """批量实验时逐个校验检索模式。"""
        if value is None:
            return value
        modes = [(x or '').lower() for x in value]
        invalid = [x for x in modes if x not in RETRIEVAL_MODES]
        if invalid:
            raise ValueError(f'存在非法retrieval_modes: {", ".join(invalid)}')
        return modes


class ContextAskReq(BaseModel):
    """上下文感知RAG问答接口的请求体（供tutor网站调用）。"""
    question: str
    algorithm_id: int | None = None
    algorithm_name: str | None = None
    session_id: str
    user_id: int | None = None
    top_k: int = 5
    retrieval_mode: str = 'hybrid'
    enable_tools: bool = True
    enable_memory: bool = True
    page_context: dict | None = None  # 可选的页面上下文信息

    @field_validator('retrieval_mode')
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        """统一校验检索模式是否合法。"""
        mode = (value or 'hybrid').lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f'retrieval_mode必须是: {", ".join(sorted(RETRIEVAL_MODES))}')
        return mode


@app.get('/')
def index():
    """返回前端页面。"""
    return FileResponse(UI_FILE)


@app.get('/health')
def health():
    """最基础的健康检查接口。"""
    return {'status': 'ok'}


@app.get('/stats')
def stats():
    """返回当前知识库规模、来源和检索模式等统计信息。"""
    try:
        return {'message': 'ok', 'data': get_system_stats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/build')
def build(req: BuildReq):
    """根据数据库内容重新构建知识库。"""
    try:
        return {'message': 'ok', 'data': build_knowledge(with_synth=req.with_synth, limit=req.limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/synthesize')
def synthesize(req: SynthesisReq):
    """触发数据合成流水线，把生成的问答写回知识库。"""
    try:
        return {
            'message': 'ok',
            'data': synthesize_dialogue_qa(
                limit=req.limit,
                algorithm_id=req.algorithm_id,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/ask')
def ask(req: AskReq):
    """系统主问答入口：规划 -> 工具/MCP -> RAG -> 直答兜底。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail='question不能为空')
    try:
        return {
            'message': 'ok',
            'data': answer_with_plan(
                req.question.strip(),
                top_k=req.top_k,
                algorithm_id=req.algorithm_id,
                retrieval_mode=req.retrieval_mode,
                current_state=req.current_state,
                enable_tools=req.enable_tools,
                enable_memory=req.enable_memory,
                enable_planning=req.enable_planning,
                enable_mcp=req.enable_mcp,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/evaluate')
def evaluate(req: EvalReq):
    """运行自动评测，对比不同问答方案。"""
    try:
        return {
            'message': 'ok',
            'data': evaluate_rag(
                limit=req.limit,
                top_k=req.top_k,
                algorithm_id=req.algorithm_id,
                retrieval_mode=req.retrieval_mode,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/search')
def search(req: SearchReq):
    """只做检索不生成答案，便于观察召回结果。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail='question不能为空')
    try:
        return {
            'message': 'ok',
            'data': search_only(
                req.question.strip(),
                top_k=req.top_k,
                algorithm_id=req.algorithm_id,
                retrieval_mode=req.retrieval_mode,
                use_expansion=req.use_expansion,
                include_memory=req.include_memory,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/memory_graph')
def memory_graph(req: MemoryReq):
    """查看当前问题命中的结构化记忆子图。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail='question不能为空')
    try:
        return {
            'message': 'ok',
            'data': select_memory_subgraph(
                req.question.strip(),
                algorithm_id=req.algorithm_id,
                top_k=req.top_k,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/experiment')
def experiment(req: ExperimentReq):
    """批量跑实验并输出最佳配置。"""
    try:
        return {
            'message': 'ok',
            'data': run_experiment_suite(
                limit=req.limit,
                algorithm_id=req.algorithm_id,
                top_k_values=req.top_k_values,
                retrieval_modes=req.retrieval_modes,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 网站集成API ====================

@app.post('/api/rag/context_ask')
def context_ask(req: ContextAskReq):
    """上下文感知的RAG问答（供tutor网站调用）
    
    核心功能:
        1. 识别用户当前浏览的算法（algorithm_id）
        2. 只在该算法范围内检索（提升精准度）
        3. 支持指代消解（"它"、"这个算法"等）
        4. 追踪会话历史（session_id）
        5. 支持连续对话
    
    请求体示例:
        {
            "question": "这个算法的时间复杂度为什么是O(nlogn)?",
            "algorithm_id": 10,
            "session_id": "user_123_session_1",
            "user_id": 123
        }
    
    返回示例:
        {
            "message": "ok",
            "data": {
                "answer": "快速排序采用分治策略...",
                "references": [...],
                "session_id": "user_123_session_1",
                "resolved_question": "快速排序的时间复杂度为什么是O(nlogn)?"
            }
        }
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail='question不能为空')
    
    try:
        result = rag_ask(
            question=req.question.strip(),
            top_k=req.top_k,
            algorithm_id=req.algorithm_id,
            session_id=req.session_id,
            retrieval_mode=req.retrieval_mode,
            enable_tools=req.enable_tools,
            enable_memory=req.enable_memory,
            user_id=req.user_id,
        )
        
        # 添加会话ID到返回结果
        result['session_id'] = req.session_id
        
        return {'message': 'ok', 'data': result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/rag/session/{session_id}')
def get_session(session_id: str, limit: int = 10):
    """获取会话历史记录
    
    参数:
        session_id: 会话ID
        limit: 返回最近N条记录（默认10）
    
    返回示例:
        {
            "message": "ok",
            "data": {
                "session_id": "user_123_session_1",
                "history": [
                    {
                        "timestamp": "2026-06-11T10:00:00",
                        "question": "快速排序的步骤是什么?",
                        "answer": "..."
                    }
                ]
            }
        }
    """
    try:
        history = get_session_history(session_id, limit=limit)
        return {
            'message': 'ok',
            'data': {
                'session_id': session_id,
                'history': history
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/rag/session/{session_id}/summary')
def get_session_info(session_id: str):
    """获取会话摘要信息
    
    返回示例:
        {
            "message": "ok",
            "data": {
                "session_id": "user_123_session_1",
                "total_messages": 5,
                "algorithms_discussed": [10, 15, 20],
                "created_at": "2026-06-11T10:00:00",
                "last_activity": "2026-06-11T10:30:00"
            }
        }
    """
    try:
        summary = get_session_summary(session_id)
        return {'message': 'ok', 'data': summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/api/rag/session/{session_id}')
def delete_session(session_id: str):
    """清空会话历史
    
    用途:
        - 用户点击"新对话"按钮
        - 清理测试数据
    """
    try:
        clear_session(session_id)
        return {
            'message': 'ok',
            'data': {'session_id': session_id, 'status': 'cleared'}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 新增功能API ====================

# 导入新模块
from .chain_visualizer import parse_reasoning_chain, create_chain_graph, extract_references_from_chain
from .advanced_planner import multi_step_plan, validate_plan
from .graph_visualizer import (
    get_graph_for_query,
    get_algorithm_subgraph,
    get_full_graph,
    get_shortest_path
)
from .conversation_manager import (
    create_branch,
    rollback_to_message,
    get_conversation_tree,
    compare_branches
)
from .quality_scorer import score_answer_quality, compare_answer_versions
from .feedback_learner import (
    record_feedback,
    get_feedback_stats,
    get_low_quality_chunks,
    generate_improvement_report
)
from .algorithm_recommender import recommend_algorithms, get_learning_path


# ========== 功能1: 可视化思维链 ==========

class ChainVisualizeReq(BaseModel):
    """思维链可视化请求"""
    answer: str


@app.post('/api/chain/visualize')
def visualize_chain(req: ChainVisualizeReq):
    """解析并可视化思维链
    
    功能：从答案中提取推理步骤，生成可视化数据
    """
    try:
        chain_data = parse_reasoning_chain(req.answer)
        graph_data = create_chain_graph(chain_data['steps'])
        references = extract_references_from_chain(req.answer)
        
        return {
            'message': 'ok',
            'data': {
                'chain': chain_data,
                'graph': graph_data,
                'references': references
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 功能2: 高级规划器 ==========

class MultiStepPlanReq(BaseModel):
    """多步规划请求"""
    question: str
    context: dict | None = None


@app.post('/api/planner/multi_step')
def plan_multi_step(req: MultiStepPlanReq):
    """多步规划：分解复杂问题
    
    功能：将复杂问题分解为多个子问题和执行步骤
    """
    try:
        plan = multi_step_plan(req.question, context=req.context)
        validation = validate_plan(plan)
        
        return {
            'message': 'ok',
            'data': {
                'plan': plan,
                'validation': validation
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 功能3: 知识图谱可视化 ==========

@app.get('/api/graph/search')
def search_graph(query: str, max_nodes: int = 20):
    """搜索知识图谱
    
    功能：根据关键词获取相关子图
    """
    try:
        graph_data = get_graph_for_query(query, max_nodes=max_nodes)
        return {'message': 'ok', 'data': graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/graph/algorithm/{algorithm_name}')
def get_algo_graph(algorithm_name: str, depth: int = 2):
    """获取算法的知识图谱
    
    功能：获取特定算法及其相关节点的子图
    """
    try:
        graph_data = get_algorithm_subgraph(algorithm_name, depth=depth)
        return {'message': 'ok', 'data': graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/graph/full')
def get_complete_graph(limit_nodes: int | None = None):
    """获取完整知识图谱
    
    警告：可能返回大量数据
    """
    try:
        graph_data = get_full_graph(limit_nodes=limit_nodes)
        return {'message': 'ok', 'data': graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/graph/path')
def find_path(from_node: str, to_node: str, max_depth: int = 5):
    """查找两个节点间的最短路径
    
    功能：用于生成学习路径
    """
    try:
        path_data = get_shortest_path(from_node, to_node, max_depth=max_depth)
        return {'message': 'ok', 'data': path_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 功能4: 对话分支管理 ==========

class CreateBranchReq(BaseModel):
    """创建分支请求"""
    session_id: str
    branch_point: int
    new_question: str


@app.post('/api/conversation/branch')
def create_conv_branch(req: CreateBranchReq):
    """创建对话分支
    
    功能：从历史某个点创建新分支
    """
    try:
        result = create_branch(req.session_id, req.branch_point, req.new_question)
        return {'message': 'ok', 'data': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RollbackReq(BaseModel):
    """回退请求"""
    session_id: str
    message_index: int


@app.post('/api/conversation/rollback')
def rollback_conversation(req: RollbackReq):
    """回退到某个历史消息
    
    功能：撤销后续对话
    """
    try:
        result = rollback_to_message(req.session_id, req.message_index)
        return {'message': 'ok', 'data': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/conversation/tree/{session_id}')
def get_conv_tree(session_id: str):
    """获取完整对话树
    
    功能：查看所有分支
    """
    try:
        tree = get_conversation_tree(session_id)
        return {'message': 'ok', 'data': tree}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 功能5: RAG质量评分 ==========

class QualityScoreReq(BaseModel):
    """质量评分请求"""
    question: str
    answer: str
    references: list[dict]


@app.post('/api/quality/score')
def score_quality(req: QualityScoreReq):
    """评估答案质量
    
    功能：实时评分答案的完整性、相关性、清晰度、引用质量
    """
    try:
        score_result = score_answer_quality(req.question, req.answer, req.references)
        return {'message': 'ok', 'data': score_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CompareAnswersReq(BaseModel):
    """对比答案请求"""
    answers: list[dict]  # [{'version': 'v1', 'answer': '...', 'references': [...]}]


@app.post('/api/quality/compare')
def compare_answers(req: CompareAnswersReq):
    """对比多个答案版本
    
    功能：找出最佳答案
    """
    try:
        comparison = compare_answer_versions(req.answers)
        return {'message': 'ok', 'data': comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 功能6: 用户反馈学习 ==========

class FeedbackReq(BaseModel):
    """用户反馈请求"""
    session_id: str
    question: str
    answer: str
    feedback_type: str  # 'thumbs_up' or 'thumbs_down'
    feedback_reason: str | None = None
    chunk_ids: list[int] | None = None


@app.post('/api/feedback/submit')
def submit_feedback(req: FeedbackReq):
    """提交用户反馈
    
    功能：点赞/点踩答案
    """
    try:
        result = record_feedback(
            req.session_id,
            req.question,
            req.answer,
            req.feedback_type,
            req.feedback_reason,
            req.chunk_ids
        )
        return {'message': 'ok', 'data': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/feedback/stats')
def feedback_statistics(chunk_id: int | None = None):
    """获取反馈统计
    
    功能：查看满意度数据
    """
    try:
        stats = get_feedback_stats(chunk_id=chunk_id)
        return {'message': 'ok', 'data': stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/feedback/low_quality')
def low_quality_chunks(threshold: float = 0.5, min_feedback: int = 3):
    """获取低质量chunk列表
    
    功能：识别需要改进的内容
    """
    try:
        chunks = get_low_quality_chunks(threshold=threshold, min_feedback=min_feedback)
        return {'message': 'ok', 'data': chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/feedback/report')
def improvement_report():
    """生成改进建议报告
    
    功能：综合分析所有反馈，给出改进建议
    """
    try:
        report = generate_improvement_report()
        return {'message': 'ok', 'data': report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 功能7: 算法推荐 ==========

class RecommendReq(BaseModel):
    """推荐请求"""
    user_id: int
    session_history: list[dict]
    top_k: int = 5


@app.post('/api/recommend/algorithms')
def recommend_algos(req: RecommendReq):
    """推荐相关算法
    
    功能：基于学习历史推荐下一步学习内容
    """
    try:
        recommendations = recommend_algorithms(
            req.user_id,
            req.session_history,
            top_k=req.top_k
        )
        return {'message': 'ok', 'data': recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/recommend/learning_path')
def learning_path(start: str, end: str):
    """生成学习路径
    
    功能：从start算法到end算法的学习路径
    """
    try:
        path = get_learning_path(start, end)
        return {'message': 'ok', 'data': path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
