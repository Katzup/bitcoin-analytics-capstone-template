# EDA: Exploratory Data Analysis

This folder contains sponsor-ready notebooks for the Bitcoin Accumulation Strategy midterm submission.

---

## 30-Second Grader Checklist

```bash
# 1. Download data (from repo root)
python download_data.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks
jupyter notebook EDA/EDA_Executive.ipynb  # Kernel → Restart & Run All
jupyter notebook EDA/EDA.ipynb            # Kernel → Restart & Run All
```

**Expected Runtime**: ~8 minutes total (~3 min executive, ~5 min full EDA)

**Expected Outputs**: 9 PNG files in `EDA/output/` (see list below)

---

## Notebooks

### EDA_Executive.ipynb
**Purpose**: Polished executive summary (~5 minute read, ~3 min runtime)

**Start Here First**

**Structure**:
1. **Executive Summary Bullets** (6 key findings with "so what")
2. **What We Believe Is True Now** (executive glue paragraph)
3. **Dataset Overview** (integrity checks + summary table)
4. **Insight Chapter 1**: The Complexity Trap (CNN vs baseline)
5. **Insight Chapter 2**: Regime Matters More Than Timing
6. **Insight Chapter 3**: The Daily Noise Problem
7. **Insight Chapter 4**: Prediction Market Exploration (Polymarket + Fear & Greed)
8. **Direction for Final Deliverable**
9. **Links to EDA.ipynb** (Sections 4, 5, 6)
10. **Compliance Checklist**

**Visuals**: ~3 plots with titles-as-claims, labeled axes, captions

---

### EDA.ipynb
**Purpose**: Comprehensive technical appendix (~5 min runtime)

**Structure**:
1. **Reproducibility Contract** (versions, paths, seeds)
2. **Section 1**: Data Loading + Sanity Checks (6 checks)
3. **Section 2**: Dataset Overview (price charts, returns, rolling stats)
4. **Section 3**: Feature Experiments (causal construction, correlation heatmap)
5. **Section 4**: Prediction Market Exploration (Polymarket + Fear & Greed substitute)
   - Polymarket column survey: 0 non-null values found across all candidate columns
   - Fear & Greed Index as substitute sentiment signal (correlation analysis)
6. **Section 5**: Robustness Checks
   - Time-window analysis (4 subperiods)
   - Bootstrap Confidence Interval (1000 resamples)
7. **Section 6**: Reproducibility Artifacts + Reusable Code Snippets

**Visuals**: ~7 plots with full statistical analysis

---

## Expected Artifacts (After Run All)

After running both notebooks, you should see exactly these files in `EDA/output/`:

```
EDA/output/
├── executive_complexity_trap.png    # Insight 1: CNN vs baseline comparison
├── executive_regimes.png            # Insight 2: Bull/bear regime analysis
├── executive_correlation.png        # Insight 3: Feature-return correlation
├── eda_price_chart.png              # Price history (linear + log)
├── eda_return_distribution.png      # Return histogram + Q-Q plot
├── eda_rolling_statistics.png       # Volatility and return clustering
├── eda_correlation.png              # Feature correlation heatmap
├── eda_fear_greed.png               # Section 4: Fear & Greed substitute analysis
├── eda_robustness.png               # Time-window robustness check
└── [bootstrap plot if generated]    # Statistical rigor CI visualization
```

**Filenames are deterministic** (no timestamps) and **overwritten on each run**.

---

## Data Requirements

Per template conventions:
- Run `python download_data.py` from repo root first
- Notebooks load from `data/*.parquet` or `data/*.csv`
- Fallback to URL if local data unavailable (for CI)

**Key Field**: `PriceUSD_coinmetrics` (BTC price in USD)

**Evaluation Window**: 2018-01-01 to 2025-12-31 (template compliant)

---

## Compliance Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Folder structure: `EDA/` | ✅ | Present |
| Filenames: `EDA_Executive.ipynb`, `EDA.ipynb` | ✅ | Present |
| Data retrieval: references `download_data.py` | ✅ | Section in both notebooks |
| Evaluation window: 2018-01-01 to 2025-12-31 | ✅ | `EVAL_START`/`EVAL_END` constants |
| No leakage: explicit forward-return definition | ✅ | Section 3 in EDA.ipynb |
| Bootstrap CI for key claim | ✅ | Section 5 in EDA.ipynb |
| Causal regime detection marked descriptive | ✅ | Explicitly noted |
| Deterministic: seed=42 | ✅ | Reproducibility contract |
| Run All compatible | ✅ | Tested |
| Output filenames deterministic | ✅ | No timestamps |

---

## Troubleshooting

### "Data file not found"
```bash
# Run from repo root, not EDA/
python download_data.py
# Then restart notebook kernel
```

### Import Errors
```python
# Ensure repo root is in path
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Plots Not Saving
```python
# Output directory is created automatically with:
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Reproducibility

| Component | Value |
|-----------|-------|
| **Seed** | 42 (deterministic across runs) |
| **Python** | 3.11+ |
| **Pandas** | 2.0+ |
| **NumPy** | 1.24+ |
| **Matplotlib** | 3.7+ |
| **Offline Capable** | Yes (after initial data download) |

---

## Track Declaration

**Track**: BTC Analytics + Accumulation Optimization

**Deliverable Type**: Tournament-ready strategy submission

**Key Design Decisions**:
- Neutral probability baseline (evidence from ablation)
- Daily allocation (tournament requirement)
- Bounded multipliers [0.7, 1.6] (risk control)
- Regime analysis descriptive only (future enhancement)

---

**Date**: February 2026  
**Project**: Stacking Sats Tournament Submission
