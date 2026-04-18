"""V1 baseline: per-pixel classifier on change-detection features.

Strategy
--------
For each training tile and each year Y in 2021..LAST:
  X = year_over_year_features(tile, Y, baseline_year=2020)
  y_pos = fused["confident_pos"] (≥2 weak sources agree post-2020 AND forest in 2020)
  y_neg = fused["confident_neg"] (no source agrees AND forest in 2020)
Sample a balanced subset of (pos, neg) pixels per tile/year and train a single
LightGBM-like model (LogReg fallback if xgboost unavailable).

Inference writes a binary deforestation raster per test tile, which
``submission_utils.raster_to_geojson`` converts to the submission GeoJSON.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio

from .data_loader import tile_grid, year_over_year_features
from .label_fusion import forest_mask_2020, fuse_post2020
from .postprocess import PostprocessConfig, postprocess

ANALYSIS_YEARS = (2021, 2022, 2023, 2024, 2025)
DEFAULT_POS_PER_TILE = 20_000
DEFAULT_NEG_PER_TILE = 20_000

try:
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except Exception:  # pragma: no cover - optional dep
    _HAS_XGB = False

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


@dataclass
class BaselineConfig:
    analysis_years: tuple[int, ...] = ANALYSIS_YEARS
    pos_per_tile: int = DEFAULT_POS_PER_TILE
    neg_per_tile: int = DEFAULT_NEG_PER_TILE
    seed: int = 42
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    xgb_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        )
    )


# ---------- sampling ----------


def _sample_pixels(
    features: np.ndarray,  # (C, H, W)
    pos_mask: np.ndarray,  # (H, W) bool
    neg_mask: np.ndarray,  # (H, W) bool
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [N, C], y [N]) drawn from the pos/neg masks."""
    pos_ij = np.argwhere(pos_mask)
    neg_ij = np.argwhere(neg_mask)
    if len(pos_ij) == 0 or len(neg_ij) == 0:
        C = features.shape[0]
        return np.zeros((0, C), dtype=np.float32), np.zeros(0, dtype=np.uint8)

    if len(pos_ij) > n_pos:
        pos_ij = pos_ij[rng.choice(len(pos_ij), n_pos, replace=False)]
    if len(neg_ij) > n_neg:
        neg_ij = neg_ij[rng.choice(len(neg_ij), n_neg, replace=False)]

    ij = np.vstack([pos_ij, neg_ij])
    y = np.concatenate([np.ones(len(pos_ij), dtype=np.uint8), np.zeros(len(neg_ij), dtype=np.uint8)])
    X = features[:, ij[:, 0], ij[:, 1]].T.astype(np.float32)
    return X, y


def build_training_set(
    train_tiles: list[str],
    cfg: BaselineConfig,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble (X, y, channel_names) from all (tile, year) in the training set."""
    rng = np.random.default_rng(cfg.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    names: list[str] | None = None

    for tile in train_tiles:
        fused = fuse_post2020(tile, split=split, gate_by_forest=True, min_sources=2)
        forest = forest_mask_2020(tile, split=split)
        for year in cfg.analysis_years:
            try:
                feats, chan_names = year_over_year_features(tile, year, split=split, baseline_year=2020)
            except FileNotFoundError:
                continue
            if names is None:
                names = chan_names

            # Pos pixels: confident_pos AND alert_date falls in this year.
            event_year = _days_to_year(fused["event_days"])
            pos = fused["confident_pos"].astype(bool) & (event_year == year)
            neg = fused["confident_neg"].astype(bool)  # no alert from any source

            X, y = _sample_pixels(feats, pos, neg, cfg.pos_per_tile, cfg.neg_per_tile, rng)
            if X.size:
                xs.append(X)
                ys.append(y)

    if not xs:
        raise RuntimeError("No training samples assembled — check label coverage / years.")

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), names or []


def _days_to_year(days_since_epoch: np.ndarray) -> np.ndarray:
    """Convert int days-since-1970 array -> year (np.int16). 0 stays 0."""
    out = np.zeros_like(days_since_epoch, dtype=np.int16)
    valid = days_since_epoch > 0
    # days -> date via np.datetime64
    dt = np.array(days_since_epoch[valid], dtype="datetime64[D]")
    out[valid] = dt.astype("datetime64[Y]").astype(int) + 1970
    return out


# ---------- model ----------


def train_model(X: np.ndarray, y: np.ndarray, cfg: BaselineConfig):
    """Train an XGBoost classifier (fallback: sklearn LogisticRegression)."""
    if _HAS_XGB:
        clf = xgb.XGBClassifier(**cfg.xgb_params)
        clf.fit(X, y)
        return clf
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Neither xgboost nor sklearn is installed") from e
    warnings.warn("xgboost not available, falling back to LogisticRegression", stacklevel=1)
    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    clf.fit(X, y)
    return clf


def predict_proba_tile(
    model,
    tile_id: str,
    year: int,
    split: str,
    chunk_rows: int = 1024,
) -> np.ndarray:
    """Apply ``model`` to a tile for one year; returns (H, W) float32 probabilities."""
    feats, _ = year_over_year_features(tile_id, year, split=split, baseline_year=2020)
    C, H, W = feats.shape
    flat = feats.reshape(C, -1).T  # (H*W, C)
    proba = np.zeros(H * W, dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_rows * W):
        end = start + chunk_rows * W
        chunk = flat[start:end]
        proba[start:end] = model.predict_proba(chunk)[:, 1].astype(np.float32)
    return proba.reshape(H, W)


def predict_proba_max(
    model,
    tile_id: str,
    split: str,
    cfg: BaselineConfig,
) -> np.ndarray:
    """Max probability across ``cfg.analysis_years`` — the raw model output per tile."""
    probs = []
    for year in cfg.analysis_years:
        try:
            p = predict_proba_tile(model, tile_id, year, split)
            probs.append(p)
        except FileNotFoundError:
            continue
    if not probs:
        g = tile_grid(tile_id, split)
        return np.zeros(g.shape, dtype=np.float32)
    return np.maximum.reduce(probs)


def predict_tile_binary(
    model,
    tile_id: str,
    split: str,
    cfg: BaselineConfig,
) -> np.ndarray:
    """Full inference + post-processing: probability → clean binary raster."""
    p_max = predict_proba_max(model, tile_id, split, cfg)
    if not p_max.any():
        return p_max.astype(np.uint8)
    forest = forest_mask_2020(tile_id, split)
    return postprocess(p_max, forest_mask=forest, config=cfg.postprocess)


# ---------- I/O ----------


def save_prediction_raster(binary: np.ndarray, tile_id: str, split: str, out_dir: Path) -> Path:
    """Write a georeferenced single-band uint8 raster aligned to the tile grid."""
    g = tile_grid(tile_id, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{tile_id}.tif"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=g.height,
        width=g.width,
        count=1,
        dtype="uint8",
        crs=g.crs,
        transform=g.transform,
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(binary, 1)
    return out_path


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(model, path)
    else:  # pragma: no cover
        import pickle

        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_model(path: Path):
    if joblib is not None:
        return joblib.load(path)
    import pickle  # pragma: no cover

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- end-to-end ----------


def train_and_predict(
    train_tiles: list[str],
    predict_tiles: list[str],
    predict_split: str = "test",
    out_dir: str | Path = "predictions/v1",
    cfg: BaselineConfig | None = None,
) -> dict:
    """Convenience driver: train on train_tiles and write prediction rasters."""
    cfg = cfg or BaselineConfig()
    out_dir = Path(out_dir)
    X, y, names = build_training_set(train_tiles, cfg, split="train")
    model = train_model(X, y, cfg)
    save_model(model, out_dir / "model.joblib")

    meta: dict = {"n_train_samples": int(X.shape[0]), "n_features": int(X.shape[1]), "channels": names, "tiles": {}}
    for tile in predict_tiles:
        binary = predict_tile_binary(model, tile, predict_split, cfg)
        raster_path = save_prediction_raster(binary, tile, predict_split, out_dir)
        meta["tiles"][tile] = {"raster": str(raster_path), "positive_fraction": float(binary.mean())}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta
