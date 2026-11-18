
import math
import pickle
from collections import Counter

from preprocess import preprocess


class InvertedIndex:
    def __init__(self):
        self.doc_ids = []       # posición interna -> id original del documento
        self.postings = {}      # término -> {posición_doc: frecuencia}
        self.doc_len = []       # nº de términos de cada doc (para BM25)
        self.doc_nterms = []    # nº de términos DISTINTOS de cada doc (para Jaccard)
        self.N = 0
        self.avgdl = 0.0
        self.tfidf_norm = []    # norma del vector TF-IDF de cada doc (para coseno)

    def build(self, docs: dict):
        """docs: {id_documento: texto crudo}"""
        for pos, (doc_id, text) in enumerate(docs.items()):
            tokens = preprocess(text)
            counts = Counter(tokens)
            self.doc_ids.append(doc_id)
            self.doc_len.append(len(tokens))
            self.doc_nterms.append(len(counts))
            for term, tf in counts.items():
                self.postings.setdefault(term, {})[pos] = tf
        self.N = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self._compute_tfidf_norms()
        return self

    def df(self, term):
        return len(self.postings.get(term, ()))

    def idf_tfidf(self, term):
        """IDF clásico para TF-IDF: log(N / df)."""
        return math.log(self.N / self.df(term))

    def idf_bm25(self, term):
        """IDF de BM25 (versión Lucene, siempre positiva)."""
        df = self.df(term)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _compute_tfidf_norms(self):
        sq = [0.0] * self.N
        for term, plist in self.postings.items():
            idf = self.idf_tfidf(term)
            for pos, tf in plist.items():
                sq[pos] += (tf * idf) ** 2
        self.tfidf_norm = [math.sqrt(x) for x in sq]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)