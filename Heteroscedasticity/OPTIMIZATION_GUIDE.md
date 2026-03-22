# LNP Encapsulation Efficiency Model - Distribution Optimization

## Problem Statement

Your XGBoost model is predicting a narrow range of EE% values (60-90%) when the actual data spans 10-99%. This is a common problem called **distribution mismatch** or **heteroscedastic prediction**.

**Visual Summary:**
```
Actual EE%:     |████████████████████████████████████| (10% to 99%, full range)
Predicted EE%:  |              ██████████              | (60% to 90%, compressed)
```

## Root Causes

1. **Training on mean values only**: Standard regression (MSE loss) optimizes for mean prediction, not distribution coverage
2. **Imbalanced target distribution**: Few extreme values (very high/low EE%) in training data
3. **Model underfitting extremes**: Model learns to predict "safe" middle values to minimize loss

## Solutions Implemented

This package implements **4 proven approaches** to match prediction distribution to actual distribution:

### ✅ Approach 1: Box-Cox Transformation

**How it works:**
- Transforms skewed target distribution into normal distribution
- Trains model on transformed data
- Inverse-transforms predictions back to original scale

**When to use:**
- Excellent for skewed distributions
- Most mathematically elegant
- Requires positive target values (handled automatically)

**Expected improvement:**
- Better coverage of extreme values
- More symmetric prediction distribution

**Code:**
```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='box-cox')
y_train_transformed = transformer.fit_transform(y_train)
# Train model on y_train_transformed
y_pred_transformed = model.predict(X_test)
y_pred = transformer.inverse_transform(y_pred_transformed)
```

---

### ✅ Approach 2: Quantile Regression

**How it works:**
- Instead of predicting mean (0.5 quantile), predict multiple quantiles (0.1, 0.25, 0.5, 0.75, 0.9)
- Model learns full distribution, not just center
- Naturally handles heteroscedasticity

**When to use:**
- Need uncertainty estimates
- Want to understand prediction at different confidence levels
- Data has non-linear relationships with variance

**Expected improvement:**
- Captures full range of values naturally
- Learns that some inputs are inherently more variable

**Code:**
```python
# Train separate models for each quantile
for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
    model = xgb.XGBRegressor(
        objective='reg:quantilehubererror',
        quantile_alpha=quantile
    )
    model.fit(X_train, y_train)
```

---

### ✅ Approach 3: Isotonic Calibration

**How it works:**
- Train standard XGBoost model
- Use separate calibration set to learn monotonic mapping: model_pred → actual_dist
- Apply this mapping to test predictions
- "Stretches" narrow prediction range to match actual range

**When to use:**
- Want to keep standard model architecture
- Need simple post-hoc fix
- Have enough data for calibration set

**Expected improvement:**
- Predictions automatically stretched to match actual distribution
- Zero additional training overhead

**Code:**
```python
from sklearn.isotonic import IsotonicRegression

# Train model, then calibrate
y_cal_pred = model.predict(X_train_cal)
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_cal_pred, y_train_cal)

# Apply to test set
y_pred_calibrated = calibrator.predict(model.predict(X_test))
```

---

### ✅ Approach 4: Weighted Loss / Sample Weighting

**How it works:**
- Give higher importance (weight) to extreme values during training
- Extreme values: very low EE% or very high EE%
- Normal values: middle EE% values
- Forces model to learn to predict full range

**When to use:**
- Model underfits extremes
- Have clear definition of "extreme"
- Want to adjust model behavior directly

**Expected improvement:**
- Better sensitivity to extreme values
- Learned features more predictive of edges

**Code:**
```python
# Weight extreme values more heavily
distances = np.abs(y_train - 50) / 50  # 0 to 1
sample_weights = 1.0 + 2.0 * distances  # 1.0 to 3.0

model.fit(X_train, y_train, sample_weight=sample_weights)
```

---

## Evaluation Metrics

To determine which approach works best, we use:

### 1. **Kolmogorov-Smirnov (KS) Test Statistic**
- **What it measures**: How different two distributions are
- **Interpretation**: 
  - **Lower is better** (0 = identical distributions)
  - p-value < 0.05 means significantly different
- **Why it matters**: Directly measures goal (match distributions)

### 2. **Percentile Coverage**
Compare actual vs predicted at key percentiles:
```
10th percentile: Actual=15%, Predicted=12%  ✓ Good
50th percentile: Actual=78%, Predicted=75%  ✓ Good
90th percentile: Actual=95%, Predicted=92%  ✓ Good
```

### 3. **Extreme Value Coverage**
Count how many predictions fall in extreme ranges:
```
<30%: Actual=45 samples, Predicted=28 samples  
>85%: Actual=120 samples, Predicted=85 samples
```

### 4. **Standard Metrics** (Secondary)
- **RMSE**: Average prediction error (in %)
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination

---

## How to Use

### Quick Start

```bash
# Run everything
python run_full_optimization.py

# Or run individual steps
python preprocess_data.py          # Loads and preprocesses data
python optimize_model_distribution.py  # Trains 4 models and compares
```

### Interpreting Output

After running `optimize_model_distribution.py`, you'll get:

1. **Console output**: Detailed metrics for each approach
2. **PNG files**: Distribution comparison plots
3. **CSV file**: Summary table of all results

**Example output:**
```
SUMMARY: APPROACH COMPARISON
                  boxcox  quantile  calibrated  weighted
RMSE             15.23%   15.89%    14.95%     16.12%
MAE              12.15%   12.67%    11.98%     12.89%
R²                0.551    0.528     0.573      0.521
KS Stat           0.142    0.118     0.095      0.165
Pred Range      10-99%    12-97%    11-98%     8-99%

✓ BEST METHOD: calibrated (KS=0.095)
```

---

## Choosing the Best Approach

| Metric | Box-Cox | Quantile | Calibrated | Weighted |
|--------|---------|----------|-----------|----------|
| **Distribution Match** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Ease of Implementation** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Interpretability** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Computational Cost** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation**: Start with **Calibrated** (Approach 3) - it's the most effective and easiest to understand.

---

## Understanding the Visualizations

Each approach generates a plot with two panels:

### Left Panel: Distribution Histograms
```
Shows frequency distribution of actual vs predicted values
Wide distribution = model predicting full range
Narrow distribution = model underfitting range
```

### Right Panel: Prediction Scatter
```
Each point is one test sample
X-axis: Actual EE%
Y-axis: Predicted EE%
Red diagonal: Perfect prediction
If points follow red line = good model
If points clustered in horizontal band = underfitting
```

---

## Advanced: Combining Approaches

You can combine techniques for even better results:

```python
# Example: Quantile regression + calibration
y_pred_quantile = train_quantile_regression(...)
calibrated_pred = calibrator.predict(y_pred_quantile)

# Example: Box-Cox + weighted loss
y_transformed = boxcox_transform(y_train)
model.fit(X_train, y_transformed, sample_weight=weights)
y_pred = boxcox_inverse(model.predict(X_test))
```

---

## Implementation in Your Workflow

### Current (Notebook):
```
Raw CSV → Preprocessing → Feature Engineering → XGBoost → Underfitting
```

### Optimized:
```
Raw CSV → Preprocessing → Feature Engineering → Choose Best Method 
    → Box-Cox / Quantile / Calibrated / Weighted → Better Coverage ✓
```

---

## Troubleshooting

**Q: My model still has narrow prediction range**
- A: The chosen method might not be the best fit
  - Check KS statistics to see which performed best
  - Try combining two approaches
  - Ensure features are properly scaled

**Q: Calibration method doesn't work**
- A: Might need more calibration data
  - Increase calibration set size (try 40% instead of 30%)
  - Use stratified split to ensure variety in calibration set

**Q: Quantile regression is very slow**
- A: Normal for multiple quantiles
  - Consider training only quantiles [0.1, 0.5, 0.9] instead of 5
  - Reduce n_estimators from 500 to 300

**Q: Box-Cox transformation gives negative predictions**
- A: Clip predictions: `np.clip(y_pred, 0, 100)`
  - This is already done in the script

---

## References

- **Box-Cox**: Box & Cox (1964) "An Analysis of Transformations"
- **Quantile Regression**: Koenker & Bassett (1978)
- **Calibration**: Platt (1999), also Niculescu-Mizil & Caruana (2005)
- **XGBoost**: Chen & Guestrin (2016)

---

## Next Steps After Optimization

1. **Validate on external data**: Test best method on new LNP formulations
2. **Feature importance analysis**: Understand which features drive EE%
3. **Sensitivity analysis**: How do individual lipids affect predictions?
4. **Production deployment**: Save best model for inference
5. **Continuous improvement**: Retrain as new experimental data arrives

---

## Questions?

If the optimized model still doesn't match your distribution:

1. Check feature quality - are SMILES features being extracted correctly?
2. Examine outliers - are there systematic biases in extreme samples?
3. Consider non-linear relationships - might need deeper trees
4. Validate data quality - ensure EE% measurements are accurate

---

**Generated**: 2024
**For**: LNP Atlas Encapsulation Efficiency Prediction
**Status**: Ready for Claude Code integration ✓
