# Model Card: XGBoost

## Model Details

| Field | Value |
|-------|-------|
| Model Name | XGBoost |
| Version | 1.0.0 |
| Date | 2025-12-15 |
| Task | Binary Classification (Churn Prediction) |
| Framework | scikit-learn / XGBoost / LightGBM |

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Training Samples | 40,000 |
| Validation Samples | 10,000 |
| Churn Rate | 39.93% |
| Features | 12 |

### Features Used

- transaction_count
- total_revenue
- avg_revenue
- std_revenue
- avg_price
- max_price
- min_price
- total_quantity
- avg_quantity
- days_since_first
- days_since_last
- avg_days_between

## Hyperparameters

```json
{
  "subsample": 1.0,
  "num_leaves": 31,
  "n_estimators": 100,
  "max_depth": 3,
  "learning_rate": 0.1
}
```

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.9990 |
| Precision | 0.9990 |
| Recall | 0.9985 |
| F1 Score | 0.9987 |
| AUC-ROC | 1.0000 |

## Intended Use

This model is intended to predict customer churn based on transaction behavior.
Use cases include:
- Identifying at-risk customers for retention campaigns
- Prioritizing customer success outreach
- Informing loyalty program targeting

## Limitations

- Trained on synthetic business events data
- Churn definition (7-day inactivity) may not align with all business contexts
- Performance may degrade with distribution shift over time
- Should be retrained periodically with fresh data

## Ethical Considerations

- Model predictions should not be the sole basis for consequential decisions
- Human review recommended for high-impact interventions
- Ensure fair treatment across customer segments

## MLflow Tracking

Experiment: `churn-classification-baselines`

---
*Generated automatically by `02_baselines.ipynb`*
