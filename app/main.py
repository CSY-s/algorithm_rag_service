"""FastAPI 接口入口。

这个文件主要负责定义请求参数，并把 HTTP 请求转发到具体业务模块。"""


import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from .evaluate import evaluate_rag
from .experiment import get_system_stats, run_experiment_suite, search_only
from .memory_graph import select_memory_subgraph
from .rag import build_knowledge
from .hybrid_retrieval import RETRIEVAL_MODES
from .agent_workflow import answer_with_plan
from .synthesis_pipeline import synthesize_dialogue_qa

app = FastAPI(title='Algorithm RAG Service', version='1.1.0')
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
