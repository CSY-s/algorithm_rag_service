"""基线方法实现模块。

用于科研对比实验，实现多种基线问答方法。"""

import re
from typing import Optional

from .db_source import fetch_algorithms
from .deepseek import deepseek_chat
from .store import get_chunks


# ============================================================================
# Baseline 1: BM25 检索 + 抽取式回答
# ============================================================================

def bm25_baseline(
    question: str,
    top_k: int = 5,
    algorithm_id: Optional[int] = None,
) -> dict:
    """BM25 检索 + 直接拼接返回（传统信息检索方法）。
    
    特点：
    - 使用 BM25 算法进行检索
    - 不使用 LLM 生成，直接返回检索片段
    - 最传统的检索式问答方法
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 获取语料
    chunks = get_chunks(algorithm_id=algorithm_id, include_memory=False)
    if not chunks:
        return {'answer': '暂无语料', 'references': [], 'method': 'bm25_baseline'}
    
    # 构建文档列表
    docs = [c['content'] for c in chunks]
    
    # TF-IDF 检索（简化版 BM25）
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        max_features=5000,
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([question])
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 获取 top_k 文档
        retrieved = [chunks[i] for i in top_indices if similarities[i] > 0]
        
        # 直接拼接返回（不使用 LLM）
        if not retrieved:
            answer = '未找到相关内容。'
        else:
            answer = '\n\n---\n\n'.join([
                f"【片段 {i+1}】\n{doc['content'][:500]}"
                for i, doc in enumerate(retrieved)
            ])
        
        return {
            'answer': answer,
            'references': retrieved,
            'method': 'bm25_baseline',
            'retrieval_count': len(retrieved),
        }
    
    except Exception as e:
        return {
            'answer': f'检索失败: {str(e)}',
            'references': [],
            'method': 'bm25_baseline',
        }


# ============================================================================
# Baseline 2: 纯 LLM 生成（无检索）
# ============================================================================

def llm_only_baseline(question: str) -> dict:
    """纯 LLM 生成，不提供任何上下文（测试 LLM 的先验知识）。
    
    特点：
    - 不进行任何检索
    - 完全依赖 LLM 的预训练知识
    - 容易产生幻觉
    """
    try:
        answer = deepseek_chat(
            [
                {
                    'role': 'system',
                    'content': '你是一个算法助教。请根据你的知识回答问题，不要编造不确定的内容。',
                },
                {'role': 'user', 'content': question},
            ],
            temperature=0.2,
            max_tokens=1600,
        )
        
        return {
            'answer': answer,
            'references': [],
            'method': 'llm_only_baseline',
            'retrieval_count': 0,
        }
    
    except Exception as e:
        return {
            'answer': f'生成失败: {str(e)}',
            'references': [],
            'method': 'llm_only_baseline',
        }


# ============================================================================
# Baseline 3: 标准 RAG（向量检索）
# ============================================================================

def vector_rag_baseline(
    question: str,
    top_k: int = 5,
    algorithm_id: Optional[int] = None,
) -> dict:
    """标准 RAG：向量检索 + LLM 生成。
    
    特点：
    - 只使用向量检索（语义相似度）
    - 不使用 TF-IDF 或关键词检索
    - 标准的 RAG 流程
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 获取语料
    chunks = get_chunks(algorithm_id=algorithm_id, include_memory=False)
    if not chunks:
        return {'answer': '暂无语料', 'references': [], 'method': 'vector_rag_baseline'}
    
    # 构建文档列表
    docs = [c['content'] for c in chunks]
    
    # 向量检索（使用 TF-IDF 作为向量表示的简化版）
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        max_features=5000,
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([question])
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 获取 top_k 文档
        retrieved = [chunks[i] for i in top_indices if similarities[i] > 0]
        
        if not retrieved:
            return {
                'answer': '未找到相关内容。',
                'references': [],
                'method': 'vector_rag_baseline',
            }
        
        # 构建上下文
        context = '\n\n'.join([
            f"[片段 {i+1}]\n{doc['content']}"
            for i, doc in enumerate(retrieved)
        ])
        
        # LLM 生成答案
        answer = deepseek_chat(
            [
                {
                    'role': 'system',
                    'content': '你是算法助教。根据提供的检索片段回答问题。',
                },
                {
                    'role': 'user',
                    'content': f"问题：{question}\n\n检索片段：\n\n{context}",
                },
            ],
            temperature=0.2,
            max_tokens=1600,
        )
        
        return {
            'answer': answer,
            'references': retrieved,
            'method': 'vector_rag_baseline',
            'retrieval_count': len(retrieved),
        }
    
    except Exception as e:
        return {
            'answer': f'生成失败: {str(e)}',
            'references': [],
            'method': 'vector_rag_baseline',
        }


# ============================================================================
# Baseline 4: 单一 TF-IDF 检索 + LLM
# ============================================================================

def tfidf_rag_baseline(
    question: str,
    top_k: int = 5,
    algorithm_id: Optional[int] = None,
) -> dict:
    """单一 TF-IDF 检索 + LLM 生成。
    
    特点：
    - 只使用 TF-IDF 检索
    - 不使用关键词或向量检索
    - 不使用知识图谱和记忆
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 获取语料
    chunks = get_chunks(algorithm_id=algorithm_id, include_memory=False)
    if not chunks:
        return {'answer': '暂无语料', 'references': [], 'method': 'tfidf_rag_baseline'}
    
    # 构建文档列表
    docs = [c['content'] for c in chunks]
    
    # TF-IDF 检索
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        max_features=5000,
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([question])
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 获取 top_k 文档
        retrieved = [chunks[i] for i in top_indices if similarities[i] > 0]
        
        if not retrieved:
            return {
                'answer': '未找到相关内容。',
                'references': [],
                'method': 'tfidf_rag_baseline',
            }
        
        # 构建上下文
        context = '\n\n'.join([
            f"[片段 {i+1}]\n{doc['content']}"
            for i, doc in enumerate(retrieved)
        ])
        
        # LLM 生成答案
        answer = deepseek_chat(
            [
                {
                    'role': 'system',
                    'content': '你是算法助教。根据提供的检索片段回答问题。',
                },
                {
                    'role': 'user',
                    'content': f"问题：{question}\n\n检索片段：\n\n{context}",
                },
            ],
            temperature=0.2,
            max_tokens=1600,
        )
        
        return {
            'answer': answer,
            'references': retrieved,
            'method': 'tfidf_rag_baseline',
            'retrieval_count': len(retrieved),
        }
    
    except Exception as e:
        return {
            'answer': f'生成失败: {str(e)}',
            'references': [],
            'method': 'tfidf_rag_baseline',
        }


# ============================================================================
# 基线方法统一接口
# ============================================================================

BASELINE_METHODS = {
    'bm25': bm25_baseline,
    'llm_only': llm_only_baseline,
    'vector_rag': vector_rag_baseline,
    'tfidf_rag': tfidf_rag_baseline,
}


def run_baseline(
    method: str,
    question: str,
    top_k: int = 5,
    algorithm_id: Optional[int] = None,
) -> dict:
    """运行指定的基线方法。
    
    Args:
        method: 基线方法名称，可选值：
            - 'bm25': BM25 检索 + 抽取式回答
            - 'llm_only': 纯 LLM 生成（无检索）
            - 'vector_rag': 标准 RAG（向量检索）
            - 'tfidf_rag': 单一 TF-IDF 检索 + LLM
        question: 用户问题
        top_k: 检索数量
        algorithm_id: 算法 ID（可选）
    
    Returns:
        包含答案、参考文档和方法名称的字典
    """
    if method not in BASELINE_METHODS:
        raise ValueError(
            f"未知的基线方法: {method}。可选值: {', '.join(BASELINE_METHODS.keys())}"
        )
    
    baseline_fn = BASELINE_METHODS[method]
    
    # llm_only 不需要 top_k 和 algorithm_id 参数
    if method == 'llm_only':
        return baseline_fn(question)
    else:
        return baseline_fn(question, top_k=top_k, algorithm_id=algorithm_id)
