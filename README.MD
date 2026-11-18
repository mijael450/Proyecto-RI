# Sistema de Recuperación de Información 

Proyecto del 1er bimestre · Recuperación de Información 

Sistema de búsqueda que compara cuatro modelos de recuperación sobre el corpus FiQA (preguntas y respuestas de finanzas, en inglés):

| Modelo | Tipo | Implementación |
|---|---|---|
| Binario | Similitud de Jaccard con vectores binarios | Propia, sobre índice invertido |
| Vectorial | Coseno con pesos TF-IDF (`tf · log(N/df)`) | Propia, sobre índice invertido |
| Probabilístico | BM25 (`k1 = 1.5`, `b = 0.75`) | Propia, sobre índice invertido |
| Vectorial semántico | Embeddings `all-MiniLM-L6-v2` + ChromaDB (distancia coseno) | `sentence-transformers` + `chromadb` |


## Despliegue

URL de la aplicación: **pendiente**

## Corpus y evaluación

- **Corpus:** [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa), 57.600 documentos (se descartan 38 textos vacíos de los 57.638 originales).
- **Consultas y qrels:** [BeIR/fiqa-qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels), split `test`: 648 consultas y 1.705 pares relevantes.
- **Métricas:** Precision@10 y Recall@10 por consulta; MAP sobre todas las consultas. La Average Precision se calcula sobre el ranking de los 100 primeros documentos y se divide entre el total de relevantes de la consulta.

### Resultados

| Modelo | MAP | Precision@10 | Recall@10 |
|---|---|---|---|
| Binario (Jaccard) | 0,0355 | 0,0148 | 0,0600 |
| Vectorial (TF-IDF) | 0,1354 | 0,0508 | 0,2159 |
| Probabilístico (BM25) | 0,2075 | 0,0704 | 0,3132 |
| Vectorial (Embeddings) | 0,3072 | 0,1042 | 0,4383 |

Cada consulta tiene en promedio unos 2,6 documentos relevantes, por lo que Precision@10 no puede superar aproximadamente 0,26.

## Estructura

```
proyecto-ri/
├── app.py                    # interfaz Streamlit
├── requirements.txt
├── data/
│   ├── corpus.jsonl          # documentos
│   ├── queries.json          # 648 consultas de prueba
│   └── qrels.json            # documentos relevantes por consulta
├── src/
│   ├── preprocess.py         # tokenización, normalización, stopwords, stemming
│   ├── inverted_index.py     # índice invertido término -> {documento: frecuencia}
│   ├── models.py             # Jaccard, TF-IDF + coseno, BM25
│   ├── embeddings.py         # embeddings y búsqueda en ChromaDB
│   └── evaluate.py           # Precision, Recall, MAP
├── scripts/
│   ├── prepare_data.py       # descarga FiQA y genera data/
│   ├── build_all.py          # índice invertido y métricas de los modelos clásicos
│   └── build_embeddings.py   # embeddings, ChromaDB y métricas del modelo semántico
└── artifacts/
    ├── index.pkl             # índice invertido
    ├── chroma_db/            # base vectorial
    └── metrics.json          # métricas precalculadas
```

## Instalación

Requiere Python 3.10 a 3.12.

```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

## Ejecución

La carpeta `data/` y `artifacts/` ya vienen en el repositorio, por lo que para usar la aplicación basta con:

```
streamlit run app.py
```

### Reconstruir todo desde cero

Ejecutar desde la raíz del proyecto, en este orden:

```
python scripts/prepare_data.py       # descarga FiQA y crea data/
python scripts/build_all.py          # índice invertido + métricas de Jaccard, TF-IDF y BM25
python scripts/build_embeddings.py   # embeddings + ChromaDB + métricas semánticas
```

`build_embeddings.py` tardó unos 55 minutos en CPU con el corpus completo. Se ejecuta una sola vez; la aplicación solo carga los resultados.

## Interfaz

- **Buscar:** escribir una consulta, elegir el modelo y ver el top-n de resultados.
- **Comparar modelos:** los cuatro rankings lado a lado y el solapamiento entre ellos.
- **Métricas:** MAP, Precision@10 y Recall@10 por modelo, y las consultas donde los embeddings mejoran o empeoran respecto a BM25.
- **Consultas de prueba:** al elegir una del selector lateral, los documentos relevantes según los qrels se marcan con ✅.

## Decisiones de diseño

- **Preprocesamiento:** minúsculas, eliminación de enlaces y entidades HTML, tokenización con expresión regular, stopwords en inglés y stemming de Porter (NLTK). Se aplica igual a documentos y consultas.
- **Índice invertido:** guarda, por término, los documentos donde aparece y su frecuencia; también la longitud de cada documento y su número de términos distintos. Los modelos clásicos solo puntúan los documentos que comparten algún término con la consulta.
- **BM25:** IDF `log(1 + (N − df + 0,5) / (df + 0,5))`, siempre positivo.
- **Embeddings:** vectores normalizados de 384 dimensiones, `max_seq_length = 256`, calculados una sola vez y guardados en ChromaDB con distancia coseno.
- **Evaluación:** solo el split `test` de los qrels.