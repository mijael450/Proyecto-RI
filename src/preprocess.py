"""Preprocesamiento de texto: tokenización, normalización y stopwords.

Se aplica EXACTAMENTE igual a documentos y consultas.
"""
import html
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def _load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOPWORDS = _load_stopwords()
STEMMER = PorterStemmer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def preprocess(text: str, stem: bool = True) -> list[str]:
    """Texto crudo -> lista de términos normalizados."""
    text = html.unescape(text)          
    text = URL_RE.sub(" ", text)        
    text = text.lower()                 
    tokens = TOKEN_RE.findall(text)    
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    return tokens