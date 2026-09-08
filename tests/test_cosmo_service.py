import numpy as np

from src.cosmo_service import build_index, normalize_text, search_similar


def test_normalize_text_removes_extra_spaces_and_lowercases():
    assert normalize_text("  UBER   BRASIL  ") == "uber brasil"


def test_build_index_and_search_returns_best_match():
    records = [
        {"merchant_name": "NETFLIX", "categoria": "Entretenimento"},
        {"merchant_name": "UBER BRASIL", "categoria": "Transporte"},
        {"merchant_name": "IFOOD", "categoria": "Alimentacao"},
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype="float32",
    )

    index, indexed_records = build_index(records, embeddings)
    result = search_similar(embeddings[0], index, indexed_records, k=1)

    assert result[0]["merchant_name"] == "NETFLIX"
    assert result[0]["categoria"] == "Entretenimento"
