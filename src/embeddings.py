"""Recuperación semántica: embeddings con sentence-transformers + ChromaDB."""
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION = "fiqa"


def load_model():
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 256      # acelera y reduce memoria
    return model


def build_collection(chroma_path, doc_ids, texts, model, batch_size=64, add_chunk=2000):
    """Calcula los embeddings de los documentos y los guarda en ChromaDB (se hace UNA vez)."""
    import chromadb
    chroma_path = Path(chroma_path).resolve()
    shutil.rmtree(chroma_path, ignore_errors=True)
    chroma_path.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(texts), add_chunk):
        chunk_ids = doc_ids[i:i + add_chunk]
        embs = model.encode(texts[i:i + add_chunk], batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=True)
        col.add(ids=chunk_ids, embeddings=embs.tolist())
        print(f"  guardados {min(i + add_chunk, len(texts))}/{len(texts)}")
    return col


def get_collection(chroma_path):
    import chromadb
    return chromadb.PersistentClient(path=str(Path(chroma_path).resolve())).get_collection(COLLECTION)


def embedding_search(col, model, query, k=10):
    """Convierte la consulta en vector y busca los k más similares. Devuelve [(doc_id, similitud)]."""
    q = model.encode([query], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[q], n_results=k)
    return [(d, 1 - dist) for d, dist in zip(res["ids"][0], res["distances"][0])]