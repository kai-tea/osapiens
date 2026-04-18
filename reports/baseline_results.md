# Baseline v1 — cross-validation results

- Git SHA: `574448a349c8cc83355773e42ad851dba5359f7e`
- Split source: `cini/splits/split_v1/fold_assignments.csv`
- Positive threshold (training target): soft_target >= 0.5
- Classifier: LightGBM binary, 402 features

## Per-fold metrics at F1-optimal threshold

| fold | threshold | precision | recall | F1 | IoU | PR-AUC | rows | positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.580 | 0.816 | 0.902 | 0.857 | 0.749 | 0.950 | 20000 | 10000 |
| 1 | 0.266 | 0.900 | 0.835 | 0.866 | 0.764 | 0.954 | 20000 | 10000 |
| 2 | 0.174 | 0.906 | 0.906 | 0.906 | 0.829 | 0.968 | 20000 | 10000 |
| **mean** |  | 0.874 | 0.881 | 0.876 | 0.781 | 0.957 |  |  |

### Submission threshold summary

- mean: 0.340
- median: 0.266
- min: 0.174
- max: 0.580

## Per-MGRS region breakdown

| region | mean F1 across folds | total rows |
| --- | ---: | ---: |
| 18NWJ | 0.906 | 20000 |
| 18NWH | 0.866 | 20000 |
| 18NWG | 0.857 | 20000 |

**Regional generalisation gap (best − worst F1)**: 18NWJ (0.906) − 18NWG (0.857) = **0.050**

## Confusion matrices per fold (at F1-optimal threshold)

| fold | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7966 | 2034 | 984 | 9016 |
| 1 | 9077 | 923 | 1652 | 8348 |
| 2 | 9064 | 936 | 936 | 9064 |

## Top-20 feature importances (total gain across folds)

| rank | feature | gain |
| --- | --- | ---: |
| 1 | `aef_e60_delta` | 151,146.2 |
| 2 | `aef_e20_delta` | 53,138.4 |
| 3 | `aef_e02_delta` | 16,215.1 |
| 4 | `aef_e36_baseline` | 15,431.8 |
| 5 | `aef_e36_delta` | 14,884.4 |
| 6 | `aef_e00_delta` | 8,401.9 |
| 7 | `aef_e05_delta` | 7,994.7 |
| 8 | `aef_e22_delta` | 6,343.9 |
| 9 | `aef_e35_delta` | 3,624.2 |
| 10 | `aef_e03_baseline` | 3,610.9 |
| 11 | `aef_e54_delta` | 3,449.4 |
| 12 | `aef_e20_baseline` | 3,352.7 |
| 13 | `aef_e01_delta` | 2,375.7 |
| 14 | `aef_e06_delta` | 2,131.4 |
| 15 | `aef_e33_delta` | 1,770.3 |
| 16 | `s2_b05_min_baseline` | 1,726.9 |
| 17 | `aef_e60_latest` | 1,590.1 |
| 18 | `aef_e12_delta` | 1,409.6 |
| 19 | `aef_e52_delta` | 1,327.2 |
| 20 | `s2_b06_mean_baseline` | 1,128.8 |

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
