from __future__ import annotations

import numpy as np

try:
    from .data import YEARS
except ImportError:
    from data import YEARS


def _safe_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    numerator = (a * b).sum(axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, eps)


def build_feature_names(embedding_dim: int) -> list[str]:
    names: list[str] = []

    for year in YEARS:
        names.extend(f"emb_{year}_{band_index}" for band_index in range(embedding_dim))

    for start_year, end_year in zip(YEARS[:-1], YEARS[1:]):
        names.extend(f"delta_{end_year}_{start_year}_{band_index}" for band_index in range(embedding_dim))

    names.extend(["cosine_2021_2020", "cosine_2022_2020", "cosine_2023_2020"])
    return names


def build_features_for_pixels(
    emb_2020: np.ndarray,
    emb_2021: np.ndarray,
    emb_2022: np.ndarray,
    emb_2023: np.ndarray,
) -> np.ndarray:
    raw_features = [emb_2020, emb_2021, emb_2022, emb_2023]
    delta_features = [
        emb_2021 - emb_2020,
        emb_2022 - emb_2021,
        emb_2023 - emb_2022,
    ]
    similarity_features = [
        _safe_cosine_similarity(emb_2021, emb_2020),
        _safe_cosine_similarity(emb_2022, emb_2020),
        _safe_cosine_similarity(emb_2023, emb_2020),
    ]

    feature_columns = raw_features + delta_features + [feature[:, None] for feature in similarity_features]
    return np.column_stack(feature_columns).astype(np.float32, copy=False)


def build_features_for_locations(
    embeddings: dict[int, np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    return build_features_for_pixels(
        embeddings[2020][:, rows, cols].T,
        embeddings[2021][:, rows, cols].T,
        embeddings[2022][:, rows, cols].T,
        embeddings[2023][:, rows, cols].T,
    )
