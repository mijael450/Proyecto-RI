import heapq
import math
from collections import Counter, defaultdict

from preprocess import preprocess


def _top_k(index, scores, k):
    best = heapq.nlargest(k, scores.items(), key=lambda x: x[1])
    return [(index.doc_ids[pos], s) for pos, s in best]


def jaccard_search(index, query, k=10):
    """Vectores binarios: J(Q, D) = |Q ∩ D| / |Q ∪ D|."""
    q_terms = set(preprocess(query))
    inter = defaultdict(int)
    for t in q_terms:
        for pos in index.postings.get(t, ()):
            inter[pos] += 1
    scores = {pos: n / (len(q_terms) + index.doc_nterms[pos] - n) for pos, n in inter.items()}
    return _top_k(index, scores, k)


def tfidf_search(index, query, k=10):
    """Coseno entre vectores TF-IDF (peso = tf * log(N/df))."""
    q_tf = Counter(t for t in preprocess(query) if t in index.postings)
    q_w = {t: tf * index.idf_tfidf(t) for t, tf in q_tf.items()}
    q_norm = math.sqrt(sum(w * w for w in q_w.values()))
    if q_norm == 0:
        return []
    dot = defaultdict(float)
    for t, wq in q_w.items():
        idf = index.idf_tfidf(t)
        for pos, tf in index.postings[t].items():
            dot[pos] += wq * (tf * idf)
    scores = {pos: d / (q_norm * index.tfidf_norm[pos])
              for pos, d in dot.items() if index.tfidf_norm[pos] > 0}
    return _top_k(index, scores, k)


def bm25_search(index, query, k=10, k1=1.5, b=0.75):
    """BM25: suma por término de idf * tf*(k1+1) / (tf + k1*(1 - b + b*|D|/avgdl))."""
    scores = defaultdict(float)
    for t in set(preprocess(query)):
        if t not in index.postings:
            continue
        idf = index.idf_bm25(t)
        for pos, tf in index.postings[t].items():
            norm = tf + k1 * (1 - b + b * index.doc_len[pos] / index.avgdl)
            scores[pos] += idf * tf * (k1 + 1) / norm
    return _top_k(index, scores, k)