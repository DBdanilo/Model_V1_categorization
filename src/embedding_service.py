from __future__ import annotations

from typing import Iterable, Sequence

from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(model: SentenceTransformer, texts: Sequence[str], *, normalize_embeddings: bool = True, show_progress_bar: bool = False):
    return model.encode(list(texts), normalize_embeddings=normalize_embeddings, show_progress_bar=show_progress_bar)
