from __future__ import annotations

from typing import Iterable, Sequence

import faiss
import numpy as np


def normalize_text(value: str) -> str:
    """Normaliza texto para comparação textual e busca semântica."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def build_index(records: Sequence[dict], embeddings: np.ndarray | None = None):
    """Cria um índice FAISS a partir de registros e embeddings."""
    if embeddings is None:
        raise ValueError("É necessário informar a matriz de embeddings para criar o índice.")

    matrix = np.asarray(embeddings, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("A matriz de embeddings deve ter formato bidimensional.")

    if len(records) != len(matrix):
        raise ValueError("Número de registros e embeddings deve ser igual.")

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, list(records)


def search_similar(query_embedding: np.ndarray, index, indexed_records: Sequence[dict], k: int = 1):
    """Busca os registros mais parecidos em um índice FAISS."""
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")

    query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    limit = min(k, len(indexed_records))
    distances, indices = index.search(query, limit)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(indexed_records):
            continue

        record = indexed_records[int(idx)]
        results.append(
            {
                "merchant_name": record.get("merchant_name"),
                "categoria": record.get("categoria"),
                "distancia": float(distance),
                "indice": int(idx),
            }
        )

    return results
