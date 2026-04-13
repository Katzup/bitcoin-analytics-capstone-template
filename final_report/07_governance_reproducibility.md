# Section 7: Governance & Reproducibility
# STATUS: DRAFT

---

## 7.1 Project Structure

The repository follows a clean separation between tournament-mode evaluation code (grader-facing) and exploratory analysis scripts (development and documentation).

```
Visual_Trading_System/
├── tournament_mode/          # Grader-facing package
│   ├── __init__.py
│   ├── evaluator.py          # Entry point; imports features + weights
│   ├── features.py           # CNN/GAF signal (torch required)
│   ├── features_simplified.py# OLS z-score signal (no torch required)
│   └── weights.py            # EMA smoother + turnover governor
├── tests/                    # 62-test suite (pytest)
├── plans/
│   └── PROJECT_STATE.md      # Authoritative record of all results
├── verify_endpoints_demo.py  # 9-check contract verification (~30 s)
├── fee_sensitivity_analysis.py
├── contrarian_signal_diagnostics.py
├── vol_amplitude_analysis.py
├── weekly_granularity_analysis.py
├── requirements.txt          # Pinned dependencies
├── Makefile                  # setup / test / demo targets
└── EXECUTION_ASSUMPTIONS.md  # Bounded-actor contract
```

The `tournament_mode/` package is the authoritative deliverable. All other scripts are analysis utilities and are not required for tournament evaluation.

---

## 7.2 Tournament Endpoint Contract

The tournament evaluator calls a single entry point:

```python
from tournament_mode.evaluator import construct_features, compute_weights

features = construct_features(price_series)   # Returns prob_up array
weights  = compute_weights(features)          # Returns daily allocation weights
```

**Contract guarantees** (verified by `verify_endpoints_demo.py`):

| Check | Guarantee |
|-------|-----------|
| weights.sum() ≈ 1.0 | Allocations are normalized (tolerance 1e-9) |
| weights.min() ≥ MIN_WEIGHT | No zero allocations (floor = 0.001) |
| len(weights) == len(prices) | Shape contract preserved |
| No future information | Rolling statistics use only past-window data |
| Deterministic output | Same inputs → same outputs (seed=42) |
| Torch-free fallback | If torch unavailable, OLS signal used automatically |

Run `python verify_endpoints_demo.py` to verify all 9 checks pass in under 30 seconds.

---

## 7.3 Torch Fallback and Backend Detection

The evaluator implements a two-backend architecture:

```python
# tournament_mode/evaluator.py
try:
    from .features import construct_features    # CNN/GAF backend (torch)
    _FEATURES_BACKEND = "features_backend: CNN"
except ImportError:
    from .features_simplified import construct_features  # OLS backend
    _FEATURES_BACKEND = "features_backend: SIMPLIFIED (torch missing)"
```

In grader environments without CUDA/torch, the OLS signal is used automatically. The active backend can be verified:

```bash
python -c "from tournament_mode.evaluator import _FEATURES_BACKEND; print(_FEATURES_BACKEND)"
```

**Important**: The tournament headline result (44.95% RW) was produced by the SIMPLIFIED (OLS) backend, which is the active implementation used in all post-midterm evaluation. The CNN backend (torch) is preserved for historical reference.

---

## 7.4 Test Suite

```bash
make test
# → pytest tests/ --ignore=tests/test_polymarket_data.py
# → 62 tests: 53 template contract tests + 9 budget/signal tests
```

Key test categories:

| Category | Count | What They Test |
|----------|-------|----------------|
| Contract | 19 | weights sum, min, shape, dtype |
| Causality | 8 | No future data leakage |
| Turnover governor | 7 | Freeze/update behavior, turnover reduction |
| Fee invariance | 4 | Mathematical proof via numerical sweep |
| Determinism | 4 | Seed reproducibility |
| Signal bounds | 9 | prob_up ∈ [0,1], ts_z clipped at ±2σ |
| **Total** | **62** | — |

All tests pass with `make test` (exit 0) in a fresh clone.

---

## 7.5 Reproducibility Instructions

### Environment Setup

```bash
# Python 3.10 or 3.11 required (tested on both)
python3.10 -m pip install -r requirements.txt

# Dependency smoke test
python -c "import pandas as pd; import pyarrow; print('dependencies ok')"
```

Key pinned dependencies:

| Package | Version constraint | Why pinned |
|---------|--------------------|------------|
| pyarrow | ≥ 23.0.0 | Parquet read compatibility |
| pandas | ≥ 1.5.0 | Rolling window behavior |
| numpy | ≥ 1.21.0 | RNG API stability |
| scikit-learn | — | OLS utilities |

### Reproducing Primary Results

```bash
# Full validation (step=1, 3,076 windows) — ~15 minutes
python3.10 trilemma_runner.py --step 1

# Fast scan (step=7, ~440 windows) — ~2 minutes
python3.10 trilemma_runner.py --step 7

# Fee sensitivity proof
python3.10 fee_sensitivity_analysis.py

# Signal diagnostics (contrarian characterization)
python3.10 contrarian_signal_diagnostics.py
```

All scripts accept `--help` for full argument documentation.

### Reproducing Ablation Results

```bash
python3.10 vol_amplitude_analysis.py       # Ablation #3
python3.10 weekly_granularity_analysis.py  # Ablation #4
```

### Smoke Test Ladder (30-second verification)

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Dependency check
python -c "import pandas as pd; import pyarrow; print('ok')"

# 3. Endpoint contract proof
python verify_endpoints_demo.py

# 4. Backend detection
python -c "from tournament_mode.evaluator import _FEATURES_BACKEND; print(_FEATURES_BACKEND)"
```

Expected output of step 4: `features_backend: SIMPLIFIED (torch missing)` in environments without torch, or `features_backend: CNN` in environments with torch installed.

---

## 7.6 Data Provenance

| Dataset | Source | Period | Notes |
|---------|--------|--------|-------|
| BTC-USD daily closes | Coinbase via yfinance | 2014–2025 | Adjusted for splits (none for BTC) |
| Evaluation windows | Computed in-harness | 2016–2025 | step=1, 3,076 windows; step=7, 440 windows |
| Polymarket data | External (not used in final signal) | — | Test file excluded from `make test` |

The price series is the sole external data dependency. No API keys, no real-time feeds, no external databases are required for evaluation.

Data is loaded deterministically from a local parquet cache (`data/btc_daily.parquet`); if absent, the evaluator fetches from yfinance and caches. The cache ensures bit-reproducible results across runs.

---

## 7.7 Bounded-Actor Contract

`EXECUTION_ASSUMPTIONS.md` documents the behavioral constraints agreed to for this evaluation:

1. **Look-ahead free**: All rolling statistics use only data available at time t. No future prices inform any allocation weight.
2. **Deterministic**: Given identical input price series and random seeds, all outputs are identical.
3. **No external state**: The evaluator maintains no persistent state between evaluation windows.
4. **Single asset**: The strategy allocates within a single 365-day BTC-USD price window; no cross-asset positions.
5. **Proportional fees**: Transaction costs enter as a uniform multiplicative factor on SPD. The fee invariance proof (Section 5.3) applies under this cost model.

These constraints are contractual, not aspirational — they are verified by the test suite and endpoint checks.

---

*All code and documentation available in the project repository. Run `make all` for full setup + test + demo in a fresh clone.*
