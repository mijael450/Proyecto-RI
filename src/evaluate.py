
def precision_at_k(ranked, relevant, k):
    """Fracción de los k primeros resultados que son relevantes."""
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked, relevant, k):
    """Fracción de los documentos relevantes que aparecen en los k primeros."""
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def average_precision(ranked, relevant):
    """Promedio de la precisión en cada posición donde aparece un relevante,
    dividido entre el total de relevantes de la consulta."""
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def evaluate(search_fn, queries, qrels, k=10, depth=100):
    """search_fn(texto, k) -> [(doc_id, score), ...]
    Devuelve métricas por consulta y MAP del sistema."""
    per_query, aps = {}, []
    for qid, text in queries.items():
        relevant = set(qrels[qid])
        ranked = [d for d, _ in search_fn(text, depth)]
        ap = average_precision(ranked, relevant)
        per_query[qid] = {
            f"precision@{k}": precision_at_k(ranked, relevant, k),
            f"recall@{k}": recall_at_k(ranked, relevant, k),
            "ap": ap,
        }
        aps.append(ap)
    return {"per_query": per_query, "MAP": sum(aps) / len(aps)}