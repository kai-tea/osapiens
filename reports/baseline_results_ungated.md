# Baseline v1 — cross-validation results (**ungated**, full-raster)

> Eval scored every pixel of each held-out tile. Pixels outside Cini's `train_mask` were labelled `0`. Numbers here track the leaderboard setup; gated results live in the sibling report.

- Git SHA: `d4dbe6271c99b1a93bcf36d12e58953965b329bc`
- Split source: `cini/splits/split_v1/fold_assignments.csv`
- Positive threshold (training target): soft_target >= 0.5
- Classifier: LightGBM binary, 402 features

## Per-fold metrics at F1-optimal threshold

| fold | threshold | precision | recall | F1 | IoU | PR-AUC | rows | positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.300 | 0.197 | 0.970 | 0.328 | 0.196 | 0.212 | 4092856 | 705804 |
| 1 | 0.846 | 0.399 | 0.620 | 0.486 | 0.321 | 0.494 | 5078516 | 779876 |
| 2 | 0.994 | 0.724 | 0.738 | 0.731 | 0.576 | 0.789 | 5037952 | 797816 |
| **mean** |  | 0.440 | 0.776 | 0.515 | 0.364 | 0.498 |  |  |

### Submission threshold summary

- mean: 0.713
- median: 0.846
- min: 0.300
- max: 0.994

## Per-MGRS region breakdown

| region | mean F1 across folds | total rows |
| --- | ---: | ---: |
| 18NWG | 0.858 | 1004004 |
| 48PWV | 0.831 | 1012036 |
| 48QVE | 0.632 | 1016064 |
| 19NBD | 0.589 | 1012036 |
| 48PYB | 0.574 | 1040400 |
| 18NXH | 0.548 | 1004004 |
| 18NWH | 0.323 | 1004004 |
| 48PUT | 0.304 | 1018072 |
| 48QWD | 0.262 | 1007012 |
| 48PXC | 0.254 | 1032256 |
| 47QMB | 0.045 | 997832 |
| 47QQV | 0.021 | 1047552 |
| 18NWJ | 0.020 | 1004004 |
| 18NXJ | 0.013 | 1008016 |
| 18NWM | 0.000 | 2008 |
| 18NYH | 0.000 | 24 |

**Regional generalisation gap (best − worst F1)**: 18NWG (0.858) − 18NWM (0.000) = **0.858**

## Confusion matrices per fold (at F1-optimal threshold)

| fold | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| 0 | 603909 | 2783143 | 21335 | 684469 |
| 1 | 3570992 | 727648 | 296059 | 483817 |
| 2 | 4015401 | 224735 | 208809 | 589007 |

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

## Per-tile sanity (ungated, on held-out fold)

| tile | region | n_pixels | train_mask pixels | positives | fraction @ proba≥0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 18NWG_6_6 | 18NWG | 1004004 | 1000000 | 400973 | 0.0000 |
| 18NWH_1_4 | 18NWH | 1004004 | 999721 | 32559 | 0.0277 |
| 18NWJ_8_9 | 18NWJ | 1004004 | 1000405 | 84262 | 0.0651 |
| 18NWM_9_4 | 18NWM | 2008 | 1092 | 13 | 0.0000 |
| 18NXH_6_8 | 18NXH | 1004004 | 1001852 | 316351 | 0.2409 |
| 18NXJ_7_6 | 18NXJ | 1008016 | 1000944 | 24442 | 0.0573 |
| 18NYH_9_9 | 18NYH | 24 | 16 | 0 | 0.0000 |
| 19NBD_4_4 | 19NBD | 1012036 | 1002971 | 91824 | 0.0877 |
| 47QMB_0_8 | 47QMB | 997832 | 6902 | 6403 | 0.9385 |
| 47QQV_2_4 | 47QQV | 1047552 | 10877 | 10551 | 0.0000 |
| 48PUT_0_8 | 48PUT | 1018072 | 102484 | 99920 | 0.8251 |
| 48PWV_7_8 | 48PWV | 1012036 | 516279 | 514165 | 0.9908 |
| 48PXC_7_7 | 48PXC | 1032256 | 146916 | 144887 | 0.0000 |
| 48PYB_3_6 | 48PYB | 1040400 | 242509 | 239222 | 0.8895 |
| 48QVE_3_0 | 48QVE | 1016064 | 173135 | 168544 | 0.9119 |
| 48QWD_2_2 | 48QWD | 1007012 | 151382 | 149380 | 0.0000 |

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
