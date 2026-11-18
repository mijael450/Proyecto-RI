
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import evaluate
from inverted_index import InvertedIndex
from models import bm25_search, jaccard_search, tfidf_search

DATA = Path("data")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)
K = 10

# --- Carga de datos ---
docs = {}
with open(DATA / "corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        docs[r["_id"]] = r["text"]
queries = json.load(open(DATA / "queries.json", encoding="utf-8"))
qrels = json.load(open(DATA / "qrels.json", encoding="utf-8"))
print(f"Documentos: {len(docs)} | Consultas: {len(queries)}")

# --- Índice invertido ---
t0 = time.time()
index = InvertedIndex().build(docs)
index.save(ART / "index.pkl")
print(f"Índice construido en {time.time() - t0:.1f}s | términos únicos: {len(index.postings)}")

# --- Evaluación ---
models = {
    "jaccard": lambda text, k: jaccard_search(index, text, k),
    "tfidf": lambda text, k: tfidf_search(index, text, k),
    "bm25": lambda text, k: bm25_search(index, text, k),
}
metrics_path = ART / "metrics.json"
metrics = json.load(open(metrics_path)) if metrics_path.exists() else {"k": K, "models": {}}
metrics["k"] = K
for name, fn in models.items():
    t0 = time.time()
    res = evaluate(fn, queries, qrels, k=K)
    pq = res["per_query"]
    metrics["models"][name] = {
        "MAP": res["MAP"],
        f"mean_precision@{K}": sum(v[f"precision@{K}"] for v in pq.values()) / len(pq),
        f"mean_recall@{K}": sum(v[f"recall@{K}"] for v in pq.values()) / len(pq),
        "per_query": pq,
    }
    m = metrics["models"][name]
    print(f"{name:8s} MAP={m['MAP']:.4f}  P@{K}={m[f'mean_precision@{K}']:.4f}  "
          f"R@{K}={m[f'mean_recall@{K}']:.4f}  ({time.time() - t0:.1f}s)")

json.dump(metrics, open(metrics_path, "w"), indent=1)
print("Guardado: artifacts/index.pkl y artifacts/metrics.json")