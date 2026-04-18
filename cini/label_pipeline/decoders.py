"""Source-specific weak-label decoders."""

from __future__ import annotations

from datetime import date

import numpy as np

from .constants import (
    ABSENT_DAYS,
    GLADS2_POST2020_OFFSET,
    RADD_POST2020_OFFSET,
    REPROJECT_NODATA,
    TARGET_START_DATE,
)


def _observed_from_arrays(*arrays: np.ndarray) -> np.ndarray:
    obs = np.zeros(arrays[0].shape, dtype=np.uint8)
    for array in arrays:
        current = np.isfinite(array) & (array != REPROJECT_NODATA)
        obs = np.maximum(obs, current.astype(np.uint8))
    return obs


def _normalize_year(year_key: int) -> int:
    year = int(year_key)
    if year < 100:
        return 2000 + year
    return year


def decode_radd(raw_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode reprojected RADD values on the target grid."""

    raw = np.asarray(raw_array)
    obs = _observed_from_arrays(raw)
    alert = np.zeros(raw.shape, dtype=np.uint8)
    conf = np.zeros(raw.shape, dtype=np.float32)
    days_since_2020 = np.full(raw.shape, ABSENT_DAYS, dtype=np.int32)

    observed_nonzero = (obs == 1) & (raw != 0)
    raw_int = raw.astype(np.int64, copy=False)
    leading_digit = raw_int // 10000
    day_offset = raw_int % 10000

    valid_codes = observed_nonzero & np.isin(leading_digit, [2, 3])
    post_2020 = valid_codes & (day_offset >= RADD_POST2020_OFFSET)

    conf[post_2020 & (leading_digit == 2)] = 0.5
    conf[post_2020 & (leading_digit == 3)] = 1.0
    alert[post_2020] = 1
    days_since_2020[post_2020] = (day_offset[post_2020] - RADD_POST2020_OFFSET).astype(np.int32)

    return obs, alert, conf, days_since_2020


def decode_gladl(
    alert_arrays_by_year: dict[int, np.ndarray],
    date_arrays_by_year: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode reprojected GLAD-L yearly alerts on the target grid."""

    if set(alert_arrays_by_year) != set(date_arrays_by_year):
        raise ValueError("GLAD-L alert and date years do not match.")
    if not alert_arrays_by_year:
        raise ValueError("GLAD-L inputs are empty.")

    years = sorted(alert_arrays_by_year)
    first_shape = next(iter(alert_arrays_by_year.values())).shape

    obs = np.zeros(first_shape, dtype=np.uint8)
    alert = np.zeros(first_shape, dtype=np.uint8)
    conf = np.zeros(first_shape, dtype=np.float32)
    days_since_2020 = np.full(first_shape, ABSENT_DAYS, dtype=np.int32)
    best_days = np.full(first_shape, np.iinfo(np.int32).max, dtype=np.int32)

    for year_key in years:
        year = _normalize_year(year_key)
        alert_array = np.asarray(alert_arrays_by_year[year_key])
        date_array = np.asarray(date_arrays_by_year[year_key])

        obs = np.maximum(obs, _observed_from_arrays(alert_array, date_array))

        valid = (
            (alert_array != REPROJECT_NODATA)
            & (date_array != REPROJECT_NODATA)
            & np.isin(alert_array, [2, 3])
            & (date_array > 0)
        )
        if not np.any(valid):
            continue

        year_start_days = (date(year, 1, 1) - TARGET_START_DATE).days
        candidate_days = year_start_days + date_array.astype(np.int32) - 1
        earlier = valid & (candidate_days < best_days)

        alert[earlier] = 1
        conf[earlier & (alert_array == 2)] = 0.5
        conf[earlier & (alert_array == 3)] = 1.0
        days_since_2020[earlier] = candidate_days[earlier]
        best_days[earlier] = candidate_days[earlier]

    return obs, alert, conf, days_since_2020


def decode_glads2(
    alert_array: np.ndarray,
    date_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode reprojected GLAD-S2 alerts on the target grid."""

    alert_values = np.asarray(alert_array)
    date_values = np.asarray(date_array)

    obs = _observed_from_arrays(alert_values, date_values)
    alert = np.zeros(alert_values.shape, dtype=np.uint8)
    conf = np.zeros(alert_values.shape, dtype=np.float32)
    days_since_2020 = np.full(alert_values.shape, ABSENT_DAYS, dtype=np.int32)

    valid = (
        (obs == 1)
        & np.isin(alert_values, [1, 2, 3, 4])
        & (date_values > 0)
        & (date_values >= GLADS2_POST2020_OFFSET)
    )

    alert[valid] = 1
    conf[valid & (alert_values == 1)] = 0.25
    conf[valid & (alert_values == 2)] = 0.5
    conf[valid & (alert_values == 3)] = 0.75
    conf[valid & (alert_values == 4)] = 1.0
    days_since_2020[valid] = (date_values[valid].astype(np.int32) - GLADS2_POST2020_OFFSET).astype(
        np.int32
    )

    return obs, alert, conf, days_since_2020
