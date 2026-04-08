"""旧版检索模块。

该文件已被 hybrid_retrieval.py 替代，保留主要是为了兼容旧调用。"""


import re
from collections import Counter
from typing import Callable

import jieba
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


RETRIEVAL_MODES = {'tfidf', 'keyword', 'vector', 'hybrid'}

_STOPWORDS = {
    '什么', '如何', '怎么', '哪些', '这个', '那个', '以及', '一个', '一种', '可以',
    '需要', '进行', '用于', '相关', '核心', '实现', '方法', '步骤', '分析', '代码',
    '问题', '答案', '请问', '一下', '算法', '思路',
}


def _kw(text: str) -> set[str]:
    """旧版关键词提取函数。"""
    return {x for x in re.findall(r'[\w\u4e00-\u9fa5]+', (text or '').lower()) if len(x) >= 2}


def _tokenize(text: str) -> list[str]:
    """旧版分词函数。"""
    raw = (text or '').strip().lower()
    if not raw:
        return []

    text = re.sub(r'[^\w\u4e00-\u9fa5]+', ' ', raw)
    tokens: list[str] = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        if tok.isdigit():
            continue
        tokens.append(tok)
    if tokens:
        return list(dict.fromkeys(tokens))
    return list(_kw(text))


def _build_query_set(
    question: str,
    use_expansion: bool,
    expansion_fn: Callable[[str], list[str]] | None,
) -> list[str]:
    """旧版查询扩展合并逻辑。"""
    queries = [question]
    if use_expansion and expansion_fn is not None:
        for q in expansion_fn(question):
            if q and q not in queries:
                queries.append(q)
    return queries


def _doc_text(row: dict) -> str:
    """旧版文档拼接函数。"""
    return f"{row.get('title', '')}\n{row.get('content', '')}".strip()


def _tfidf_scores(rows: list[dict], queries: list[str]) -> list[float]:
    """旧版 TF-IDF 打分。"""
    docs = [_doc_text(r) for r in rows]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    mat = vec.fit_transform(docs + queries)
    doc_mat = mat[: len(docs)]
    query_mat = mat[len(docs) :]
    sim_matrix = cosine_similarity(doc_mat, query_mat)
    return np.max(sim_matrix, axis=1).astype(float).tolist()


def _vector_scores(rows: list[dict], queries: list[str]) -> list[float]:
    """旧版向量式打分。"""
    docs = [_doc_text(r) for r in rows]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1, max_features=4000)
    mat = vec.fit_transform(docs + queries)
    if mat.shape[0] < 3 or mat.shape[1] < 3:
        return _tfidf_scores(rows, queries)

    max_components = min(128, mat.shape[0] - 1, mat.shape[1] - 1)
    if max_components < 2:
        return _tfidf_scores(rows, queries)

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    dense = svd.fit_transform(mat)
    dense = normalize(dense)
    doc_vecs = dense[: len(docs)]
    query_vecs = dense[len(docs) :]
    sim_matrix = cosine_similarity(doc_vecs, query_vecs)
    return np.max(sim_matrix, axis=1).astype(float).tolist()


def _keyword_scores(rows: list[dict], queries: list[str]) -> list[float]:
    """旧版关键词打分。"""
    query_tokens = _tokenize(' '.join(queries))
    if not query_tokens:
        return [0.0] * len(rows)

    query_counter = Counter(query_tokens)
    scores: list[float] = []
    for row in rows:
        title = str(row.get('title', '') or '')
        content = str(row.get('content', '') or '')
        doc_tokens = _tokenize(f'{title} {content}')
        doc_counter = Counter(doc_tokens)
        overlap = sum(min(doc_counter[t], query_counter[t]) for t in query_counter)
        coverage = overlap / max(1, len(query_tokens))
        title_hits = sum(1 for t in query_tokens if t in title.lower())
        exact_hits = len(query_counter.keys() & _kw(f'{title} {content}'))
        score = 0.55 * coverage + 0.25 * (title_hits / max(1, len(query_tokens))) + 0.20 * (
            exact_hits / max(1, len(query_counter))
        )
        scores.append(float(score))
    return scores


def _memory_bonus(row: dict) -> float:
    """旧版 memory 加分逻辑。"""
    return 0.05 if row.get('chunk_type') == 'memory' else 0.0


def _normalize(scores: list[float]) -> list[float]:
    """旧版分数归一化逻辑。"""
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    mx = float(arr.max())
    mn = float(arr.min())
    if mx - mn < 1e-12:
        return [0.0 if mx <= 0 else 1.0 for _ in scores]
    return ((arr - mn) / (mx - mn)).astype(float).tolist()


def retrieve_rows(
    rows: list[dict],
    question: str,
    top_k: int = 5,
    mode: str = 'hybrid',
    use_expansion: bool = True,
    include_memory: bool = True,
    expansion_fn: Callable[[str], list[str]] | None = None,
) -> list[dict]:
    """旧版统一检索入口。"""
    if not rows:
        return []

    selected_mode = (mode or 'hybrid').lower()
    if selected_mode not in RETRIEVAL_MODES:
        selected_mode = 'hybrid'

    queries = _build_query_set(question, use_expansion=use_expansion, expansion_fn=expansion_fn)
    tfidf_scores = _tfidf_scores(rows, queries)
    keyword_scores = _keyword_scores(rows, queries)
    vector_scores = _vector_scores(rows, queries)

    if selected_mode == 'tfidf':
        base_scores = tfidf_scores
    elif selected_mode == 'keyword':
        base_scores = keyword_scores
    elif selected_mode == 'vector':
        base_scores = vector_scores
    else:
        tfidf_norm = _normalize(tfidf_scores)
        keyword_norm = _normalize(keyword_scores)
        vector_norm = _normalize(vector_scores)
        base_scores = []
        for i, row in enumerate(rows):
            score = 0.35 * tfidf_norm[i] + 0.25 * keyword_norm[i] + 0.40 * vector_norm[i]
            if include_memory:
                score += _memory_bonus(row)
            base_scores.append(float(score))

    if selected_mode != 'hybrid' and include_memory:
        base_scores = [float(score + _memory_bonus(rows[i])) for i, score in enumerate(base_scores)]

    idx = np.argsort(np.array(base_scores))[::-1][: max(1, top_k)]
    out = []
    for i in idx:
        item = dict(rows[int(i)])
        item['score'] = round(float(base_scores[int(i)]), 6)
        item['retrieval_mode'] = selected_mode
        item['retrieval_scores'] = {
            'tfidf': round(float(tfidf_scores[int(i)]), 6),
            'keyword': round(float(keyword_scores[int(i)]), 6),
            'vector': round(float(vector_scores[int(i)]), 6),
        }
        out.append(item)
    return out
