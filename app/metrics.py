"""评测指标模块。

实现 ROUGE、BLEU 等文本相似度指标。"""


import math
import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    """把文本切成用于评测的基本 token。"""
    text = (text or '').lower()
    return re.findall(r'[\u4e00-\u9fa5]|[a-z0-9_]+', text)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """构造 n-gram 序列。"""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _f1(overlap: int, pred_cnt: int, ref_cnt: int) -> float:
    """根据重叠数计算 F1。"""
    if pred_cnt == 0 or ref_cnt == 0 or overlap == 0:
        return 0.0
    p = overlap / pred_cnt
    r = overlap / ref_cnt
    return 2 * p * r / (p + r)


def rouge_n(pred: str, ref: str, n: int = 1) -> float:
    """计算 ROUGE-N。"""
    pt = _tokens(pred)
    rt = _tokens(ref)
    png = Counter(_ngrams(pt, n))
    rng = Counter(_ngrams(rt, n))
    overlap = 0
    for g, c in png.items():
        overlap += min(c, rng.get(g, 0))
    return _f1(overlap, sum(png.values()), sum(rng.values()))


def _lcs_len(a: list[str], b: list[str]) -> int:
    """计算两个 token 序列的最长公共子序列长度。"""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def rouge_l(pred: str, ref: str) -> float:
    """计算 ROUGE-L。"""
    pt = _tokens(pred)
    rt = _tokens(ref)
    lcs = _lcs_len(pt, rt)
    return _f1(lcs, len(pt), len(rt))


def bleu4(pred: str, ref: str) -> float:
    """计算 BLEU-4。"""
    pt = _tokens(pred)
    rt = _tokens(ref)
    if not pt:
        return 0.0

    max_n = 4
    precisions = []
    for n in range(1, max_n + 1):
        png = Counter(_ngrams(pt, n))
        rng = Counter(_ngrams(rt, n))
        overlap = 0
        for g, c in png.items():
            overlap += min(c, rng.get(g, 0))
        # add-1 smoothing
        p = (overlap + 1.0) / (sum(png.values()) + 1.0)
        precisions.append(p)

    log_p = sum((1.0 / max_n) * math.log(p) for p in precisions)

    ref_len = len(rt)
    pred_len = len(pt)
    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (ref_len / max(1, pred_len)))

    return bp * math.exp(log_p)


def score_answer(pred: str, ref: str) -> dict:
    """一次性计算一组常用评测指标。"""
    return {
        'rouge_1': round(rouge_n(pred, ref, 1), 6),
        'rouge_2': round(rouge_n(pred, ref, 2), 6),
        'rouge_l': round(rouge_l(pred, ref), 6),
        'bleu_4': round(bleu4(pred, ref), 6),
    }


def mean_scores(items: list[dict]) -> dict:
    """对多条样本分数取平均。"""
    if not items:
        return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0, 'bleu_4': 0.0}

    keys = ['rouge_1', 'rouge_2', 'rouge_l', 'bleu_4']
    out = {}
    for k in keys:
        out[k] = round(sum(float(x.get(k, 0.0)) for x in items) / len(items), 6)
    return out
