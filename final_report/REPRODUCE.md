# One-Command Reproducibility Guide
# VTS / Trilemma Stacking-SATS Practicum | Bob Katz | Spring 2026

## Quick Start (< 5 minutes)

```bash
# 1. Install dependencies
python3.10 -m pip install -r requirements.txt

# 2. Smoke test (confirms environment is correct)
python3.10 -c "import pandas as pd; import pyarrow; print('deps ok')"

# 3. Contract verification (30-second proof)
python3.10 verify_endpoints_demo.py

# 4. Confirm backend mode (CNN vs simplified)
python3.10 -c "from tournament_mode.evaluator import _FEATURES_BACKEND; print(_FEATURES_BACKEND)"
```

Expected output of step 4: `features_backend: SIMPLIFIED` (torch not required; simplified OLS backend active).

## Reproducing Official Results

### Headline result: +3.01 pp over neutral DCA

The full step=1 evaluation (3,076 windows, ~30–60 min) that produced the headline 44.95% RW:

```bash
# NOTE: The evaluator runs this internally during grader evaluation.
# To reproduce exactly:
python3.10 -c "
import pandas as pd, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from tournament_mode import construct_features
from tournament_mode.evaluator import evaluate_rolling_windows
from tournament_mode.scoring import calculate_recency_weighted_percentile

DATA_URL = 'https://raw.githubusercontent.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet'
df = pd.read_parquet(DATA_URL).loc['2016-01-01':'2025-06-01']
df_feat = construct_features(df.copy())
results = evaluate_rolling_windows(df_feat, model=None, window_days=365, step_days=1, verbose=False)
rw = calculate_recency_weighted_percentile(results['pct_strategy'].values, decay=0.9)
print(f'RW Percentile: {rw:.2f}%  (expect ~44.95%)')
"
```

### Fast scan (step=7, ~2 min) — approximate only, not the headline

```bash
python3.10 -c "
# same as above but step_days=7
# Expected: ~45.84% RW (fast-scan approximation)
"
```

## Artifacts in This Repo

| File | What it proves |
|------|---------------|
| `verify_endpoints_demo.py` | Contract: sum=1, min≥MIN_WEIGHT, deterministic, governor backward-compat |
| `fee_sensitivity_analysis.py` | Mathematical proof that proportional fees cancel in RW percentile |
| `contrarian_signal_diagnostics.py` | Signal characterization: regime-relative dampening (not classically contrarian) |
| `vol_amplitude_analysis.py` | Vol-amplitude ablation: null result → do not adopt |
| `weekly_granularity_analysis.py` | Weekly resampling ablation: null result → do not adopt |
| `tournament_mode/weights.py` | Turnover governor implementation (freeze_then_ema) |
| `tournament_mode/evaluator.py` | Torch fallback + _FEATURES_BACKEND audit variable |
| `plans/PROJECT_STATE.md` | Full experiment log with results, decisions, and rationale |

## Seed / Determinism

All runs use `RANDOM_SEED = 42`. The signal and allocator are deterministic given the same data.

## Notes for Graders

- Python interpreter: `python3.10`
- No API keys required
- Data fetched from public GitHub URL (no local data files required)
- `torch` is NOT required; the system auto-falls back to `features_simplified` (OLS backend)
