# Evaluation summary

Best model (by test F1, stress class): **random_forest**

- **random_forest** — test accuracy: 0.736, test F1 (stress class): 0.253, test F1 (macro): 0.546, 5-fold CV F1: 0.320 +/- 0.009
- **xgboost** — test accuracy: 0.713, test F1 (stress class): 0.229, test F1 (macro): 0.526, 5-fold CV F1: 0.327 +/- 0.030

## Vegetation-stress incidence by country (all 2019-2024 periods)

- Netherlands: 21.3%
- Afghanistan: 21.3%
- New Zealand: 21.3%

## Per-country test-set performance (best model)

- Netherlands: n=325, stress_rate=18.8%, accuracy=0.738, F1(stress)=0.158
- Afghanistan: n=507, stress_rate=25.6%, accuracy=0.746, F1(stress)=0.332
- New Zealand: n=393, stress_rate=15.8%, accuracy=0.720, F1(stress)=0.214