"""Constants used by the weak-label pipeline."""

from __future__ import annotations

from datetime import date

REPROJECT_NODATA = -9999.0
OUTPUT_NODATA = -9999.0
UNDEFINED_FLOAT = -1.0
ABSENT_DAYS = -1
AMBIGUOUS_HARD_LABEL = 255

RADD_BASE_DATE = date(2014, 12, 31)
GLADS2_BASE_DATE = date(2019, 1, 1)
TARGET_START_DATE = date(2020, 1, 1)

RADD_POST2020_OFFSET = (TARGET_START_DATE - RADD_BASE_DATE).days
GLADS2_POST2020_OFFSET = (TARGET_START_DATE - GLADS2_BASE_DATE).days

REGION_FIELD_CANDIDATES = (
    "region_id",
    "region",
    "country",
    "country_code",
    "mgrs_grid",
)

LABELPACK_BAND_NAMES = [
    "train_mask",
    "hard_label",
    "seed_mask",
    "soft_target",
    "sample_weight",
    "obs_count",
    "radd_obs",
    "radd_alert",
    "radd_conf",
    "radd_days_since_2020",
    "gladl_obs",
    "gladl_alert",
    "gladl_conf",
    "gladl_days_since_2020",
    "glads2_obs",
    "glads2_alert",
    "glads2_conf",
    "glads2_days_since_2020",
]
