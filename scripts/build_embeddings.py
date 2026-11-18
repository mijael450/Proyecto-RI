import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embeddings import build_collection, embedding_search, load_model
from evaluate import evaluate

DATA, ART = Path("data"), Path("artifacts")
K = 10

ids, texts = [], []
with open(DATA / "corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        ids.append(r["_id"])
        texts.append(r["text"])
queries = json.load(open(DATA / "queries.json", encoding="utf-8"))
qrels = json.load(open(DATA / "qrels.json", encoding="utf-8"))

model = load_model()
t0 = time.time()
col = build_collection(ART / "chroma_db", ids, texts, model)
print(f"Embeddings listos en {(time.time() - t0) / 60:.1f} min")

res = evaluate(lambda text, k: embedding_search(col, model, text, k), queries, qrels, k=K)
pq = res["per_query"]
entry = {
    "MAP": res["MAP"],
    f"mean_precision@{K}": sum(v[f"precision@{K}"] for v in pq.values()) / len(pq),
    f"mean_recall@{K}": sum(v[f"recall@{K}"] for v in pq.values()) / len(pq),
    "per_query": pq,
}
print(f"embeddings MAP={entry['MAP']:.4f}  P@{K}={entry[f'mean_precision@{K}']:.4f}  "
      f"R@{K}={entry[f'mean_recall@{K}']:.4f}")

path = ART / "metrics.json"
metrics = json.load(open(path)) if path.exists() else {"k": K, "models": {}}
metrics["models"]["embeddings"] = entry
json.dump(metrics, open(path, "w"), indent=1)
print("metrics.json actualizado")