# Model Comparison Report - Task 2

## Credit Card Dataset

### Performance Summary

| Model | F1-Score | Precision | Recall | ROC-AUC | PR-AUC | Cost/10K |
|-------|----------|-----------|--------|---------|--------|----------|
| Logistic Regression | 0.1141 | 0.0608 | 0.9184 | 0.9721 | 0.7189 | $14700.00 |
| Random Forest | 0.7534 | 0.6720 | 0.8571 | 0.9801 | 0.8220 | $1810.00 |
| XGBoost | 0.8586 | 0.8817 | 0.8367 | 0.9682 | 0.8800 | $1710.00 |
| XGBoost (Optimized) | 0.8586 | 0.8817 | 0.8367 | 0.9682 | 0.8800 | $1710.00 |

## Cross-Validation Results (5-fold)

| Model | CV F1-Score (Mean) | CV F1-Score (Std) |
|-------|-------------------|------------------|
| Logistic Regression | 0.1175 | ±0.0060 |
| Random Forest | 0.7778 | ±0.0301 |
| XGBoost | 0.8589 | ±0.0071 |

## Model Selection Justification

**Selected Model**: XGBoost

**Reasons:**
1. **Highest F1-Score**: 0.8586 (balanced metric)
2. **Best Precision-Recall Balance**: 0.8817 / 0.8367
3. **Lowest Business Cost**: $1710.00 per 10,000 transactions
4. **Robust CV Performance**: 0.8589 (±0.0071)

### Improvement over Baseline
- **652.7% higher F1-Score** than Logistic Regression
- **$12990.00 cost savings** per 10,000 transactions