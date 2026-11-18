"""Interfaz web (Streamlit) del sistema de recuperación de información sobre FiQA."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inverted_index import InvertedIndex  # noqa: E402
from models import bm25_search, jaccard_search, tfidf_search  # noqa: E402

DATA, ART = ROOT / "data", ROOT / "artifacts"

MODELS = {
    "Binario (Jaccard)": "jaccard",
    "Vectorial (TF-IDF)": "tfidf",
    "Probabilístico (BM25)": "bm25",
    "Vectorial (Embeddings)": "embeddings",
}
LABEL = {v: k for k, v in MODELS.items()}

st.set_page_config(page_title="Recuperación de Información - FiQA", layout="wide")


# ---------- Carga de recursos (una sola vez) ----------
@st.cache_resource(show_spinner="Cargando índice...")
def load_index():
    return InvertedIndex.load(ART / "index.pkl")


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_embedder():
    """Devuelve (modelo, colección) o None si no se han generado los embeddings."""
    if not (ART / "chroma_db").exists():
        return None
    from embeddings import get_collection, load_model  # import tardío: es pesado
    return load_model(), get_collection(ART / "chroma_db")


@st.cache_data(show_spinner="Cargando documentos...")
def load_corpus():
    docs = {}
    with open(DATA / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs[r["_id"]] = r["text"]
    return docs


@st.cache_data
def load_eval_data():
    queries = json.load(open(DATA / "queries.json", encoding="utf-8"))
    qrels = json.load(open(DATA / "qrels.json", encoding="utf-8"))
    return queries, qrels


@st.cache_data
def load_metrics():
    path = ART / "metrics.json"
    return json.load(open(path)) if path.exists() else None


index, corpus = load_index(), load_corpus()
test_queries, qrels = load_eval_data()
metrics = load_metrics()
text_to_qid = {t.strip(): q for q, t in test_queries.items()}


def run_search(model_key, query, k):
    if model_key == "jaccard":
        return jaccard_search(index, query, k)
    if model_key == "tfidf":
        return tfidf_search(index, query, k)
    if model_key == "bm25":
        return bm25_search(index, query, k)
    emb = load_embedder()
    if emb is None:
        return None
    from embeddings import embedding_search
    return embedding_search(emb[1], emb[0], query, k)


def show_results(results, relevant=frozenset()):
    if results is None:
        st.warning("Los embeddings aún no se han generado (ejecuta scripts/build_embeddings.py).")
        return
    if not results:
        st.info("Sin resultados: ningún término de la consulta aparece en el corpus.")
        return
    for rank, (doc_id, score) in enumerate(results, start=1):
        mark = "✅ " if doc_id in relevant else ""
        with st.expander(f"{rank}. {mark}Doc {doc_id} · puntaje {score:.4f}"):
            text = corpus[doc_id]
            st.write(text[:800] + ("..." if len(text) > 800 else ""))


# ---------- Barra lateral: consulta ----------
NONE = "(escribir mi propia consulta)"
if "query" not in st.session_state:
    st.session_state["query"] = ""


def _pick_example():
    sel = st.session_state["example"]
    if sel != NONE:
        st.session_state["query"] = test_queries[sel]


st.sidebar.header("Consulta")
st.sidebar.selectbox(
    "Consulta de prueba (con qrels)", [NONE] + list(test_queries), key="example",
    on_change=_pick_example,
    format_func=lambda q: q if q == NONE else f"{q}: {test_queries[q][:60]}",
)
st.sidebar.text_area("Texto de la consulta", key="query", height=110)
top_n = st.sidebar.slider("Top-n resultados", 3, 20, 10)

query = st.session_state["query"].strip()
relevant = set(qrels.get(text_to_qid.get(query, ""), {}))
if relevant:
    st.sidebar.success(f"Consulta de prueba: {len(relevant)} documento(s) relevante(s). Se marcan con ✅.")

st.title("Sistema de Recuperación de Información")
st.caption(f"Corpus FiQA · {index.N:,} documentos · Modelos: Jaccard, TF-IDF, BM25 y Embeddings")

tab_search, tab_compare, tab_metrics = st.tabs(["🔎 Buscar", "⚖️ Comparar modelos", "📊 Métricas"])

# ---------- Pestaña: buscar ----------
with tab_search:
    choice = st.selectbox("Modelo de recuperación", list(MODELS))
    if not query:
        st.info("Escribe una consulta en la barra lateral.")
    else:
        show_results(run_search(MODELS[choice], query, top_n), relevant)

# ---------- Pestaña: comparar ----------
with tab_compare:
    if not query:
        st.info("Escribe una consulta en la barra lateral.")
    else:
        results_by_model = {name: run_search(key, query, top_n) for name, key in MODELS.items()}
        cols = st.columns(len(MODELS))
        for col, (name, res) in zip(cols, results_by_model.items()):
            with col:
                st.subheader(name)
                show_results(res, relevant)
        # Solapamiento entre rankings (top-n)
        ids = {n: {d for d, _ in r} for n, r in results_by_model.items() if r}
        if len(ids) > 1:
            names = list(ids)
            overlap = pd.DataFrame(
                [[len(ids[a] & ids[b]) for b in names] for a in names], index=names, columns=names)
            st.markdown(f"**Documentos en común en el top-{top_n}** entre cada par de modelos:")
            st.dataframe(overlap)

# ---------- Pestaña: métricas ----------
with tab_metrics:
    if metrics is None:
        st.warning("No se encontró artifacts/metrics.json (ejecuta scripts/build_all.py).")
    else:
        k = metrics["k"]
        rows = []
        for key, m in metrics["models"].items():
            rows.append({"Modelo": LABEL[key], "MAP": m["MAP"],
                         f"Precision@{k}": m[f"mean_precision@{k}"],
                         f"Recall@{k}": m[f"mean_recall@{k}"]})
        df = pd.DataFrame(rows).set_index("Modelo")
        st.markdown(f"Evaluación sobre **{len(test_queries)} consultas de prueba** (split test de FiQA).")
        st.dataframe(df.style.format("{:.4f}"))
        st.bar_chart(df["MAP"])

        models_pq = {key: m["per_query"] for key, m in metrics["models"].items()}
        if "embeddings" in models_pq and "bm25" in models_pq:
            st.subheader("Embeddings vs. BM25 por consulta (Average Precision)")
            diff = pd.DataFrame([
                {"qid": q, "Consulta": test_queries[q][:100],
                 "AP BM25": models_pq["bm25"][q]["ap"],
                 "AP Embeddings": models_pq["embeddings"][q]["ap"]}
                for q in test_queries if q in models_pq["bm25"] and q in models_pq["embeddings"]])
            diff["Diferencia"] = diff["AP Embeddings"] - diff["AP BM25"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Donde los embeddings MEJORAN más**")
                st.dataframe(diff.nlargest(10, "Diferencia"), hide_index=True)
            with c2:
                st.markdown("**Donde los embeddings EMPEORAN más**")
                st.dataframe(diff.nsmallest(10, "Diferencia"), hide_index=True)