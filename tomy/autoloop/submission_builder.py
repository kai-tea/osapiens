"""Final submission file construction + leaderboard.md validation.

Single responsibility: read the autoloop's candidate GeoJSONs + their CV
scores, pick one according to a deterministic rule, rewrite it in the
exact shape leaderboard.md demands, and validate before writing.

Leaderboard.md requirements (copied here so this module is self-
contained; check ~/osapiens/leaderboard.md for the authoritative text):

  - file extension must be .geojson
  - GeoJSON must parse and use a top-level FeatureCollection
  - only Polygon and MultiPolygon are accepted
  - time_step may be valid YYMM, null, or omitted
  - YYMM means two-digit year + two-digit month, e.g. 2204 = April 2022

Everything else we write (tile_id, candidate tag, etc.) is stripped —
the leaderboard only reads geometry + optional time_step, and extra
properties risk rejection or confuse the grader.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class LeaderboardValidationError(ValueError):
    """Raised when a submission file violates leaderboard.md requirements."""


VALID_GEOMETRY_TYPES = ("Polygon", "MultiPolygon")


def _is_valid_yymm(value) -> bool:
    """True if value is an int in the shape YYMM with MM in 1-12.

    Accepts plain ints; rejects strings, floats, None, negatives.
    YYMM range here: 2001–9912 (two-digit year 00-99, month 01-12).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    if value < 1 or value > 9912:
        return False
    month = value % 100
    return 1 <= month <= 12


def validate_geojson(path: Path) -> dict:
    """Raise LeaderboardValidationError on any spec violation. Return a report.

    Cheap; opens the file once, parses it, walks every feature, returns a
    short dict with counts. Safe to call after writing and before
    declaring a run "successful".
    """
    if path.suffix != ".geojson":
        raise LeaderboardValidationError(f"{path}: extension must be .geojson")

    try:
        payload = json.loads(path.read_text())
    except Exception as err:
        raise LeaderboardValidationError(f"{path}: unparseable JSON ({err})") from err

    if not isinstance(payload, dict):
        raise LeaderboardValidationError(f"{path}: top-level must be an object")
    if payload.get("type") != "FeatureCollection":
        raise LeaderboardValidationError(
            f"{path}: top-level type must be 'FeatureCollection', got {payload.get('type')!r}"
        )

    features = payload.get("features")
    if not isinstance(features, list):
        raise LeaderboardValidationError(f"{path}: 'features' must be a list")
    if len(features) == 0:
        raise LeaderboardValidationError(f"{path}: no features — grader rejects empty submissions")

    n_with_ts = 0
    n_without_ts = 0
    per_type: dict[str, int] = {}
    for i, feat in enumerate(features):
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            raise LeaderboardValidationError(
                f"{path}: feature {i} is not a GeoJSON Feature"
            )
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            raise LeaderboardValidationError(f"{path}: feature {i} has no geometry")
        gtype = geom.get("type")
        if gtype not in VALID_GEOMETRY_TYPES:
            raise LeaderboardValidationError(
                f"{path}: feature {i} geometry type {gtype!r} not in {VALID_GEOMETRY_TYPES}"
            )
        per_type[gtype] = per_type.get(gtype, 0) + 1

        props = feat.get("properties")
        if props is None:
            props = {}
            feat["properties"] = props
        elif not isinstance(props, dict):
            raise LeaderboardValidationError(
                f"{path}: feature {i} properties is not an object"
            )

        if "time_step" in props:
            ts = props["time_step"]
            if ts is None:
                n_without_ts += 1
            elif _is_valid_yymm(ts):
                n_with_ts += 1
            else:
                raise LeaderboardValidationError(
                    f"{path}: feature {i} time_step={ts!r} is not valid YYMM"
                )
        else:
            n_without_ts += 1

    return {
        "path": str(path),
        "n_features": len(features),
        "n_with_time_step": n_with_ts,
        "n_without_time_step": n_without_ts,
        "geometry_types": per_type,
    }


def sanitise_geojson_in_place(path: Path) -> None:
    """Rewrite the file so properties only contain an optional time_step.

    Strips tile_id, debug fields, NaN-valued time_step, anything else.
    Converts missing/invalid time_step to "omitted" (property absent).
    Safe to run on already-clean files — it's a no-op in that case.
    """
    payload = json.loads(path.read_text())
    for feat in payload.get("features", []):
        props = feat.get("properties") or {}
        ts = props.get("time_step")
        clean_props: dict = {}
        if _is_valid_yymm(ts):
            clean_props["time_step"] = ts
        feat["properties"] = clean_props
    path.write_text(json.dumps(payload))


# -------- submission selection ------------------------------------


@dataclass
class Candidate:
    tag: str
    path: Path
    cv_f1: float | None
    kind: str  # "heuristic" | "model" | "ensemble"


def _load_candidates(summary_json: Path, submission_root: Path) -> list[Candidate]:
    payload = json.loads(summary_json.read_text())
    candidates: list[Candidate] = []
    for h in payload.get("heuristics", []):
        p = submission_root.parent.parent / h.get("combined_geojson", "")
        if p.exists():
            candidates.append(Candidate(h["tag"], p, None, "heuristic"))
    for m in payload.get("model_stages", []):
        p = submission_root.parent.parent / m.get("combined_geojson", "")
        if not p.exists():
            continue
        kind = "ensemble" if "ensemble" in m["tag"] or "any_signal" in m["tag"] else "model"
        candidates.append(Candidate(m["tag"], p, m.get("cv_f1_mean"), kind))
    return candidates


def select_final_candidate(
    summary_json: Path, submission_root: Path, hansen_anchor_tag: str = "hansen_tc25_ly22to25_ts"
) -> Candidate:
    """Pick the submission candidate per this rule:

    1. Among model candidates with CV F1, take the best.
    2. If the best model CV F1 beats the Hansen anchor's CV F1 (or we
       have no Hansen CV F1 to compare against) by ≥ 0.02, use it.
    3. Otherwise prefer ``any_signal`` ensemble if present (three
       independent signals hedge Cini-label bias).
    4. Otherwise fall back to ``hansen_tc25_ly22to25_ts`` (safe anchor
       — same signal as the 31.2% baseline plus per-polygon time_step).
    5. Otherwise fall back to any heuristic that exists.

    Never returns a candidate whose file is empty or missing.
    """
    cands = _load_candidates(summary_json, submission_root)
    if not cands:
        raise LeaderboardValidationError(
            "no candidate GeoJSONs found; autoloop produced nothing shippable"
        )

    # Best model (config or self-trained), must actually have a CV score.
    models = [c for c in cands if c.kind == "model" and c.cv_f1 is not None]
    best_model = max(models, key=lambda c: c.cv_f1) if models else None

    # Hansen-anchor reference — not all autoloop runs compute its CV F1
    # (naive CV is costly), so it may be None.
    hansen_anchor = next((c for c in cands if c.tag == hansen_anchor_tag), None)

    if best_model is not None:
        threshold = (hansen_anchor.cv_f1 + 0.02) if (hansen_anchor and hansen_anchor.cv_f1) else 0.0
        if best_model.cv_f1 >= threshold:
            logger.info(
                "picked model candidate %s (CV F1 %.3f, threshold %.3f)",
                best_model.tag, best_model.cv_f1, threshold,
            )
            return best_model

    any_signal = next((c for c in cands if c.tag == "any_signal"), None)
    if any_signal is not None:
        logger.info("picked any_signal ensemble (hedges Cini-label risk)")
        return any_signal

    if hansen_anchor is not None:
        logger.info("picked Hansen anchor %s (safe fallback)", hansen_anchor.tag)
        return hansen_anchor

    fallback = cands[0]
    logger.warning("no preferred candidate available — falling back to %s", fallback.tag)
    return fallback


# -------- end-to-end ---------------------------------------------


def build_final_submission(
    summary_json: Path,
    submission_root: Path,
    output_path: Path,
) -> dict:
    """Pick → sanitise → validate → write ``output_path``. Return a report."""
    candidate = select_final_candidate(summary_json, submission_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate.path, output_path)
    sanitise_geojson_in_place(output_path)
    report = validate_geojson(output_path)
    report["chosen_candidate"] = candidate.tag
    report["chosen_candidate_path"] = str(candidate.path)
    report["chosen_cv_f1"] = candidate.cv_f1
    (output_path.with_suffix(".validation.json")).write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    logger.info("final submission -> %s (from %s)", output_path, candidate.tag)
    return report
