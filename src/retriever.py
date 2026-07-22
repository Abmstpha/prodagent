"""Production retrieval pipeline (Lab 1): hybrid BM25 + dense + RRF -> parent-child -> cross-encoder rerank."""
import math
import re
from collections import Counter

from .corpus import CORPUS, CHILDREN, PARENTS, CHILD_TO_PARENT


def _tokenise(text):
    return re.findall(r"[a-zàâçéèêëîïôûùüÿœ0-9]+", text.lower())


class TinyTfidf:
    "Minimal TF-IDF retriever (cosine), no external dependency."
    def __init__(self, docs: dict):
        self.names = list(docs.keys())
        self.texts = list(docs.values())
        tok = [_tokenise(t) for t in self.texts]
        N = len(self.texts)
        df = Counter(w for ts in tok for w in set(ts))
        self.idf = {w: math.log((1 + N) / (1 + df[w])) + 1 for w in df}
        self.vecs = [self._vec(ts) for ts in tok]

    def _vec(self, tokens):
        tf = Counter(tokens)
        n = len(tokens) or 1
        return {w: tf[w] / n * self.idf.get(w, 0.0) for w in tf}

    @staticmethod
    def _cos(a, b):
        num = sum(a[w] * b[w] for w in set(a) & set(b))
        na = math.sqrt(sum(v ** 2 for v in a.values()))
        nb = math.sqrt(sum(v ** 2 for v in b.values()))
        return num / (na * nb) if na * nb else 0.0

    def search(self, query: str, k: int = 3) -> list:
        qv = self._vec(_tokenise(query))
        scores = [(self._cos(qv, v), t) for v, t in zip(self.vecs, self.texts)]
        return sorted(scores, reverse=True)[:k]


class TinyBM25:
    "Minimal BM25-Okapi, no external dependency."
    def __init__(self, docs: dict, k1: float = 1.5, b: float = 0.75):
        self.names = list(docs.keys())
        self.texts = list(docs.values())
        self.tok = [_tokenise(t) for t in self.texts]
        N = len(self.texts)
        avgdl = sum(len(ts) for ts in self.tok) / N
        df = Counter(w for ts in self.tok for w in set(ts))
        self.idf = {w: math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1) for w in df}
        self._k1 = k1
        self._b = b
        self._avgdl = avgdl

    def _score(self, tok_q, idx):
        dl = len(self.tok[idx])
        tf_d = Counter(self.tok[idx])
        sc = 0.0
        for w in tok_q:
            if w not in self.idf:
                continue
            f = tf_d[w]
            num = self.idf[w] * f * (self._k1 + 1)
            den = f + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            sc += num / den
        return sc

    def search(self, query: str, k: int = 3) -> list:
        q = _tokenise(query)
        ranked = sorted(range(len(self.texts)), key=lambda i: self._score(q, i), reverse=True)
        return [(self._score(q, i), self.texts[i]) for i in ranked[:k]]


def rrf_fusion(lists: list, K: int = 60) -> list:
    """Reciprocal Rank Fusion — fuses several lists of (score, text)."""
    scores = {}
    for lst in lists:
        for rank, (_, text) in enumerate(lst):
            scores[text] = scores.get(text, 0.0) + 1.0 / (K + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def cross_encoder_score(query: str, document: str) -> float:
    """Lightweight cross-encoder proxy (term coverage + context-richness bonus).
    Swap for sentence_transformers.CrossEncoder in a GPU deployment."""
    q_tok = set(_tokenise(query))
    d_tok = set(_tokenise(document))
    if not q_tok or not d_tok:
        return 0.0
    overlap = len(q_tok & d_tok) / len(q_tok)
    doc_bonus = min(1.0, len(d_tok) / 50)
    return round(overlap * 0.7 + doc_bonus * 0.3, 4)


def rerank(query: str, candidates: list, top_k: int = 3) -> list:
    scored = [(cross_encoder_score(query, doc), doc) for doc in candidates]
    return [doc for _, doc in sorted(scored, reverse=True)[:top_k]]


_baseline = TinyTfidf(CORPUS)
_dense_children = TinyTfidf(CHILDREN)
_bm25_children = TinyBM25(CHILDREN)


def baseline_retrieve(query: str, k: int = 3) -> list:
    """Baseline for the RAGAS table: plain TF-IDF over whole documents."""
    return [text for _, text in _baseline.search(query, k)]


def production_retrieve(query: str, k_final: int = 3) -> list:
    """Full pipeline: hybrid (dense+BM25+RRF) over children -> parents -> rerank.

    Dense leg = Pinecone (llama-text-embed-v2) when PINECONE_API_KEY is set,
    plus the local TF-IDF proxy — both fused; offline the local leg carries alone.
    """
    from .pinecone_dense import pinecone_search
    dense_c = _dense_children.search(query, k=10)
    bm25_c = _bm25_children.search(query, k=10)
    lists = [dense_c, bm25_c]
    pinecone_c = pinecone_search(query, k=10)
    if pinecone_c:
        lists.append(pinecone_c)
    fused_c = rrf_fusion(lists)
    seen, candidates = set(), []
    for text, _ in fused_c:
        child_id = next((cid for cid, ct in CHILDREN.items() if ct == text), None)
        if child_id:
            pid = CHILD_TO_PARENT[child_id]
            if pid not in seen:
                seen.add(pid)
                candidates.append(PARENTS[pid])
    return rerank(query, candidates, top_k=k_final)
