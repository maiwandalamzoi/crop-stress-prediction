# Evaluation summary

Best model (by test F1, stress class): **xgboost**

- **random_forest** — test accuracy: 0.704, test F1 (stress class): 0.404, test F1 (macro): 0.603, best CV F1 (tuning): 0.437, decision threshold: 0.48
- **xgboost** — test accuracy: 0.722, test F1 (stress class): 0.405, test F1 (macro): 0.612, best CV F1 (tuning): 0.454, decision threshold: 0.50

## Vegetation-stress incidence by country (all 2019-2024 periods)

- Netherlands: 21.3%
- Afghanistan: 21.3%
- New Zealand: 21.3%

## Per-country test-set performance (best model)

- Netherlands: n=325, stress_rate=18.8%, accuracy=0.655, F1(stress)=0.188
- Afghanistan: n=507, stress_rate=25.6%, accuracy=0.813, F1(stress)=0.625
- New Zealand: n=393, stress_rate=15.8%, accuracy=0.659, F1(stress)=0.264