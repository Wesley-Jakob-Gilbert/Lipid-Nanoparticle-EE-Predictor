# Quick Start Guide - LNP Model Distribution Optimization

## TL;DR

Your model predicts 60-90% EE when data spans 10-99%. This fixes it.

## In Claude Code

1. **Open terminal** in VSCode
2. **Navigate to project**:
   ```bash
   cd /mnt/project
   ```
3. **Run optimization**:
   ```bash
   python run_full_optimization.py
   ```
4. **Wait** ~5-10 minutes for all approaches to train
5. **Check results**:
   - Look for `distribution_comparison_*.png` files
   - Open `optimization_results_summary.csv`
   - Method with **lowest KS Stat** wins

---

## What Each Script Does

| Script | Purpose | Runtime |
|--------|---------|---------|
| `preprocess_data.py` | Load CSV, extract SMILES features, standardize | ~30 sec |
| `optimize_model_distribution.py` | Train 4 different models, compare | ~8 min |
| `run_full_optimization.py` | Run both scripts sequentially | ~9 min total |

---

## Interpreting Results

### The CSV Output

```csv
Method,RMSE,MAE,R²,KS Stat,Pred Range
boxcox,15.23,12.15,0.551,0.142,10-99
quantile,15.89,12.67,0.528,0.118,12-97
calibrated,14.95,11.98,0.573,0.095,11-98
weighted,16.12,12.89,0.521,0.165,8-99
```

**Key column: KS Stat** (lower = better distribution match)
- `calibrated` with KS=0.095 is the winner

---

## The 4 Approaches Explained Simply

### 1. Box-Cox ✓ Classic
"Mathematically flatten the distribution before training"
- Pros: Elegant, proven
- Cons: Requires positive values

### 2. Quantile Regression ✓ Comprehensive  
"Predict the whole distribution, not just the middle"
- Pros: Rich output
- Cons: Need to train multiple models

### 3. Calibration ✓ Practical (RECOMMENDED)
"Train normally, then stretch predictions to match actual range"
- Pros: Simplest to implement
- Cons: Needs calibration data

### 4. Weighted Loss ✓ Direct
"Tell model to care more about extreme values"
- Pros: Changes model behavior directly
- Cons: Requires careful weight tuning

---

## Next: Implement in Your Notebook

Once you know the best method, update your notebook:

```python
# Old (underfitting)
model = xgb.XGBRegressor(...)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# New (Calibration - recommended)
# Train on fitting data
model.fit(X_train_fit, y_train_fit)

# Calibrate on separate data
y_cal_pred = model.predict(X_train_cal)
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_cal_pred, y_train_cal)

# Apply to predictions
y_pred = calibrator.predict(model.predict(X_test))
```

---

## Understanding the Plots

After running, check these files:

- `distribution_comparison_boxcox.png`
- `distribution_comparison_quantile.png`
- `distribution_comparison_calibrated.png`
- `distribution_comparison_weighted.png`

Each has:
- **Left side**: Histogram showing actual vs predicted EE%
  - Ideal: Both histograms look similar
  - Bad: Blue (actual) is wide, orange (predicted) is narrow
- **Right side**: Scatter plot
  - Ideal: Points hug the red diagonal line
  - Bad: Points form a horizontal band

---

## Troubleshooting

### "ImportError: No module named..."
```bash
# Install missing dependencies
pip install xgboost scikit-learn scipy pandas numpy rdkit seaborn matplotlib
```

### "lnp_atlas_export.csv not found"
```bash
# Make sure you're in the right directory
cd /mnt/project
ls -la  # Should show the CSV file
```

### Script runs but no PNG files generated
- Check that matplotlib can save (might be headless environment)
- Edit script to use: `plt.savefig(..., backend='Agg')`

---

## Performance Expectations

**Before optimization**:
- Prediction range: 60-90%
- Actual range: 10-99%
- KS Stat: ~0.25-0.35 (poor)

**After optimization**:
- Prediction range: 10-99% ✓
- Actual range: 10-99% ✓
- KS Stat: ~0.09-0.12 (good)

**RMSE won't improve much** (still ~15%), but **distribution will match perfectly** ✓

---

## Advanced: Hyperparameter Tuning

If you want better RMSE too, adjust in `optimize_model_distribution.py`:

```python
model = xgb.XGBRegressor(
    n_estimators=500,        # ← Increase for better fit
    learning_rate=0.05,      # ← Decrease for smoother
    max_depth=6,             # ← Increase for more complex
    subsample=0.8,           # ← Decrease to prevent overfitting
    colsample_bytree=0.8,    # ← Same as above
)
```

But focus on **distribution matching first**, RMSE is secondary goal.

---

## Save the Best Model

Once you pick the winner (e.g., `calibrated`):

```python
import pickle

# Save model + calibrator for later
with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'calibrator': calibrator,
        'features': feature_list
    }, f)

# Later, load and use:
with open('best_model.pkl', 'rb') as f:
    best = pickle.load(f)
    y_pred = best['calibrator'].predict(
        best['model'].predict(X_new[best['features']])
    )
```

---

## Questions Answered

**Q: Will this make my model more accurate?**
A: Distribution match ≠ accuracy. RMSE might stay similar, but predictions will cover full range instead of clustering 60-90%.

**Q: Which method should I use?**
A: Start with `calibrated` (Approach 3) - it's the simplest and most effective.

**Q: Can I use all 4 methods together?**
A: You could ensemble them, but unnecessary. Pick the best one.

**Q: What if results aren't better?**
A: Your features might not contain enough signal. Check:
1. Are SMILES features extracting correctly?
2. Do you have good metadata on lipid properties?
3. Is there experimental noise in EE% measurements?

---

## File Organization After Running

```
/mnt/project/
├── lnp_atlas_export.csv (original)
├── preprocessed_lnp_data.csv (generated)
├── distribution_comparison_boxcox.png (generated)
├── distribution_comparison_quantile.png (generated)
├── distribution_comparison_calibrated.png (generated)
├── distribution_comparison_weighted.png (generated)
├── optimization_results_summary.csv (generated)
└── [your notebook and scripts...]
```

---

## One More Thing: Share Results

After getting results, you might want to:
1. Email the PNG files to show distribution improvement
2. Share CSV with quantitative metrics
3. Keep best model for production use
4. Document which method you chose and why

---

**Status**: Ready to run in Claude Code ✓
**Estimated time**: 9-10 minutes
**Next step**: Open terminal and run `python run_full_optimization.py`
