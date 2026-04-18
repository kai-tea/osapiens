# Baseline v1 — cross-validation results

- Git SHA: `0f31a530cf5ec5c3bec4d0c5dcf0ffef48a6859f`
- Split source: `cini/splits/split_v1/fold_assignments.csv`
- Positive threshold (training target): soft_target >= 0.5
- Classifier: LightGBM binary, 402 features

## Per-fold metrics at F1-optimal threshold

| fold | threshold | precision | recall | F1 | IoU | PR-AUC | rows | positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.326 | 0.966 | 0.980 | 0.973 | 0.948 | 0.992 | 311985 | 248091 |
| 1 | 0.315 | 0.956 | 0.956 | 0.956 | 0.916 | 0.988 | 500000 | 240139 |
| 2 | 0.663 | 0.971 | 0.978 | 0.974 | 0.950 | 0.996 | 406902 | 214243 |
| **mean** |  | 0.964 | 0.971 | 0.968 | 0.938 | 0.992 |  |  |

### Submission threshold summary

- mean: 0.435
- median: 0.326
- min: 0.315
- max: 0.663

## Per-MGRS region breakdown

| region | mean F1 across folds | total rows |
| --- | ---: | ---: |
| 48PWV | 0.998 | 100000 |
| 48QWD | 0.993 | 100000 |
| 48PYB | 0.993 | 100000 |
| 48PXC | 0.993 | 100000 |
| 48QVE | 0.986 | 100000 |
| 47QQV | 0.985 | 10877 |
| 48PUT | 0.977 | 100000 |
| 47QMB | 0.954 | 6902 |
| 18NWG | 0.873 | 100000 |
| 18NXH | 0.858 | 100000 |
| 19NBD | 0.814 | 100000 |
| 18NWJ | 0.684 | 100000 |
| 18NWH | 0.587 | 100000 |
| 18NXJ | 0.561 | 100000 |
| 18NWM | 0.000 | 1092 |
| 18NYH | 0.000 | 16 |

**Regional generalisation gap (best − worst F1)**: 48PWV (0.998) − 18NWM (0.000) = **0.998**

## Confusion matrices per fold (at F1-optimal threshold)

| fold | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| 0 | 55367 | 8527 | 4874 | 243217 |
| 1 | 249409 | 10452 | 10619 | 229520 |
| 2 | 186325 | 6334 | 4662 | 209581 |

## Top-20 feature importances (total gain across folds)

| rank | feature | gain |
| --- | --- | ---: |
| 1 | `aef_e12_baseline` | 3,726,591.4 |
| 2 | `aef_e52_latest` | 1,733,780.8 |
| 3 | `aef_e36_baseline` | 391,544.0 |
| 4 | `aef_e06_delta` | 376,151.2 |
| 5 | `aef_e22_delta` | 323,413.1 |
| 6 | `aef_e12_latest` | 309,713.1 |
| 7 | `aef_e01_baseline` | 220,046.0 |
| 8 | `aef_e20_delta` | 208,918.3 |
| 9 | `aef_e21_latest` | 143,827.9 |
| 10 | `aef_e02_delta` | 53,251.3 |
| 11 | `aef_e36_delta` | 44,064.7 |
| 12 | `aef_e60_delta` | 35,434.7 |
| 13 | `aef_e22_baseline` | 30,865.5 |
| 14 | `aef_e30_latest` | 30,760.4 |
| 15 | `aef_e06_latest` | 28,254.8 |
| 16 | `aef_e01_delta` | 26,502.4 |
| 17 | `aef_e03_baseline` | 25,015.5 |
| 18 | `aef_e15_baseline` | 24,746.6 |
| 19 | `aef_e15_latest` | 19,149.7 |
| 20 | `aef_e43_latest` | 16,099.1 |

## Training configuration

```json
{
  "bagging_fraction": 0.8,
  "bagging_freq": 5,
  "early_stopping_rounds": 30,
  "feature_fraction": 0.9,
  "internal_val_fraction": 0.2,
  "is_unbalance": true,
  "learning_rate": 0.05,
  "min_child_samples": 100,
  "n_estimators": 500,
  "n_jobs": -1,
  "num_leaves": 63,
  "reg_alpha": 0.0,
  "reg_lambda": 0.1,
  "seed": 42
}
```
