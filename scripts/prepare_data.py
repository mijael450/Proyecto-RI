import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_DOCS = None            
OUT = Path("data")
OUT.mkdir(exist_ok=True)

print("Descargando FiQA...")
corpus = load_dataset("BeIR/fiqa", "corpus")["corpus"]
queries = load_dataset("BeIR/fiqa", "queries")["queries"]
qrels_test = load_dataset("BeIR/fiqa-qrels", split="test")

# Documentos no vacíos
docs = {str(r["_id"]): r["text"] for r in corpus if r["text"].strip()}
q_text = {str(r["_id"]): r["text"] for r in queries}

# qrels del split test
qrels = {}
for r in qrels_test:
    if int(r["score"]) > 0 and str(r["corpus-id"]) in docs:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
qrels = {q: d for q, d in qrels.items() if q in q_text}

# Selección de documentos
relevant_ids = {d for rel in qrels.values() for d in rel}
if N_DOCS is None:
    subset_ids = sorted(docs)            
else:
    others = sorted(set(docs) - relevant_ids)
    random.Random(SEED).shuffle(others)
    n_fill = max(0, N_DOCS - len(relevant_ids))
    subset_ids = sorted(relevant_ids | set(others[:n_fill]))

with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
    for d in subset_ids:
        f.write(json.dumps({"_id": d, "text": docs[d]}, ensure_ascii=False) + "\n")

test_queries = {q: q_text[q] for q in qrels}
json.dump(test_queries, open(OUT / "queries.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(qrels, open(OUT / "qrels.json", "w", encoding="utf-8"))

print(f"Documentos en el corpus: {len(subset_ids)}")
print(f"  (de ellos, relevantes para el test: {len(relevant_ids)})")
print(f"Consultas de prueba: {len(test_queries)}")
print(f"Pares relevantes (qrels): {sum(len(v) for v in qrels.values())}")