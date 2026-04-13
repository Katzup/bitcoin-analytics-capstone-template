# PROJECT_STATE — Visual Trading System (VTS) / Trilemma Stacking SATS Practicum

Last updated: 2026-03-04 (weekly granularity experiment — null result)
Owner: Bob Katz (GT OMSA/Practicum)

## 1) Current Objective
Deliver a midterm-ready, template-compliant implementation of the required tournament endpoints:
- `construct_features(df) -> df_enriched`
- `compute_weights(df_window) -> pd.Series`

Primary goal for midterm: **working, deterministic, grader-safe endpoints + proof** (not necessarily beating DCA).

---

## 2) Canonical Endpoints Contract (Grader Interface)

### Required imports (clean path)
```python
from tournament_mode import construct_features, compute_weights, MIN_WEIGHT
```

### `construct_features(df)`

**Contract**
- Input: `pd.DataFrame` with at least `PriceUSD_coinmetrics` and `DatetimeIndex` (sorted, no dupes)
- Output: same DataFrame columns preserved + adds `prob_up` column
- Must be causal: row `t` features use only data <= `t`
- Must preserve index exactly

**Current behavior**
- OLS contrarian + z-score shrinkage: `prob_up ∈ [0.35, 0.65]`, NaN for first 200 rows (insufficient history)
- Achieves 44.95% RW percentile, 66.94% win rate (vs neutral 41.94% / 70.42%)

### `compute_weights(df_window)`

**Contract**
- Input: window-sized DataFrame (typically 365 rows) containing `prob_up` (and may contain price)
- Output: `pd.Series` indexed exactly like `df_window.index`
- Constraints:
  - `weights.sum() == 1.0` (tight tolerance)
  - `weights.min() >= MIN_WEIGHT`
  - Deterministic across repeated runs and across two grader workflows:
    - A) `construct_features(full_df)` -> slice -> `compute_weights(window)`
    - B) `construct_features(window_df)` -> `compute_weights(window)`

**NaN handling**
- If `prob_up` contains NaN (e.g., during lookback), fill deterministically:
  - `prob_up = prob_up.fillna(0.5)`

---

## 3) Current Signal and Performance

**Signal: OLS Regime-Relative + Z-Score Shrinkage**
```
TrendScore = 0.4*(slope_60d × R²_60d) + 0.6*(slope_180d × R²_180d)  [annualized]
ts_z       = clip((TrendScore - rolling_mean_252d) / rolling_std_252d, -2, +2)
prob_up    = 0.5 - 0.15 × tanh(ts_z)   ← INVERTED (buys less in extended uptrends)
```

**Performance ladder (full step=1 validation, 3076 windows, 2016–2025)**:

| Signal                       | RW%    | Win%   | Δ vs neutral |
|------------------------------|--------|--------|--------------|
| CNN (GAF)                    | 41.43% | 54.32% | −0.51 pp     |
| Neutral (constant 0.5)       | 41.94% | 70.42% | baseline     |
| OLS contrarian (raw tanh)    | 43.25% | 91.48% | +1.31 pp     |
| **OLS winsorized (z-clip)** | **44.95%** | **66.94%** | **+3.01 pp** |

**Key insights**:
- The ts_z signal is **regime-relative** (not classically contrarian): it measures trend strength
  relative to the prior 252 days, not absolute price level.
- Diagnostic result: ts_z is **positively correlated** with forward returns (r ≈ +0.14 at 30d),
  i.e., momentum-like. Q1 (bearish) does NOT reliably correspond to cheapest sats in absolute terms.
- Correct framing: the inversion reduces allocation during *extended* uptrends (high ts_z relative
  to recent history). The z-score normalization dampens tilt in high-volatility regimes (same
  absolute trend → smaller tilt when recent std is high).
- Annual win rate vs DCA: ~50% (4/8 years), consistent with marginal/mixed per-year edge.
- RW percentile is the PRIMARY tournament metric; win rate is reported but not the optimization target.

---

## 3b) Turnover Governor + Fee Robustness

### Turnover Governor (Mode A: freeze_then_ema)

Implemented in `tournament_mode/weights.py` as `_ema_with_governor()`.

**Parameters** (defaults in `compute_weights()`):
```python
gov_enabled        = True    # Mode A: sequential freeze_then_ema
gov_thresh_l1      = 0.04   # L1 freeze threshold (multiplier units)
gov_thresh_maxdelta= 0.02   # max-element freeze threshold
# Effective freeze: delta < min(thresh_l1, thresh_maxdelta) = 0.02
```

**Mechanism**: Replaces vectorized pandas EMA with day-by-day loop. At each
day t ≥ 1, if |target[t] − smoothed[t-1]| < 0.02, carry forward (freeze).
Otherwise apply EMA: α × target + (1−α) × smoothed[t-1].

**Results** (real BTC data 2020–2022, step=7 fast scan):
- Freeze rate: ~62% (226/364 days frozen on a typical window)
- Turnover reduction: ~12% vs vectorized EMA
- RW% impact: zero (45.84% both with/without governor)
- Contract compliance: sum=1 and min≥MIN_WEIGHT verified at both gov states

**Correct framing**: The EMA (α=0.30) already provides most within-window damping.
The governor adds marginal operational stability — a safety net against day-to-day
weight churn when the signal barely changes, without affecting long-run performance.

### Fee / Execution Sensitivity Analysis

**Key result**: The RW percentile is **insensitive to proportional fees in this
evaluation harness** (both strategy and DCA scale by the same fee factor, so
the tournament percentile is mathematically unchanged).

Mathematical proof: with fee factor `f = 1 + fee_bps/10_000`:
- `dynamic_spd_fee = dynamic_spd / f`
- `uniform_spd_fee = uniform_spd / f`
- `min_spd_fee = min_spd / f`,  `max_spd_fee = max_spd / f`
- `pct_fee = (dynamic_spd/f − min_spd/f) / (max_spd/f − min_spd/f) = pct_nofee`

Note: RW rank is stable across fee levels; **absolute** SPD advantage shrinks
slightly (both parties pay the fee) but remains positive at all tested levels.

**Numerical sweep** (step=7, 440 windows, 2016–2025):

| fee_bps | RW%   | Win%   | Strat SPD  | DCA SPD    | Advantage | FeeDrag   | Result    |
|---------|-------|--------|------------|------------|-----------|-----------|-----------|
| 0       | 45.84 | 66.14% | 22,274.9   | 21,970.1   | +304.9    | 0.0       | beats_DCA |
| 10      | 45.84 | 66.14% | 22,252.7   | 21,948.1   | +304.6    | -22.3     | beats_DCA |
| 25      | 45.84 | 66.14% | 22,219.4   | 21,915.3   | +304.1    | -55.5     | beats_DCA |
| 50      | 45.84 | 66.14% | 22,164.1   | 21,860.7   | +303.3    | -110.8    | beats_DCA |

SPD units: sats per dollar (1e8 × weight/price, mean over 440 windows).
Absolute SPD advantage at 50 bps: +303.3 (vs +304.9 at 0 bps) — fee drag is ~0.5%.

**Conclusion**: The +3.01 pp RW percentile advantage is stable across 0–50 bps
because the fee factor cancels in the tournament percentile formula. Both strategy
and DCA are impacted equally; the rank advantage is preserved.

Script: `fee_sensitivity_analysis.py`

---

## 3c) Vol-Normalized Amplitude Experiment

**Hypothesis**: Scaling the tilt amplitude inversely with realized vol (explicit
amplitude = clip(0.15 / vol_ratio, 0.06, 0.22)) should improve accumulation in
high-vol regimes (where current strategy may be too aggressive) and low-vol regimes
(where it may be too conservative).

**Key question**: Does this explicit vol-normalization add anything beyond what
ts_z's implicit normalization (via rolling 252d std denominator) already provides?

**Note**: the explicit layer uses a short-horizon realized-vol ratio (rv_60/rv_ref),
while ts_z standardizes against a long-horizon (252d) scale; the experiment tests
whether the additional short-horizon adjustment improves outcomes.

**Result** (step=7 coarse scan, 440 windows, 2016–2025):

| Metric                         | Fixed (0.15) | Vol-Normalized | Delta      |
|-------------------------------|--------------|----------------|------------|
| RW Percentile (primary)        | 45.84%       | 45.99%         | +0.14 pp   |
| Win Rate vs DCA                | 66.14%       | 64.55%         | −1.59%     |
| Mean Strat SPD (sats/$)        | 22,274.9     | 22,242.9       | −32.0      |
| Absolute SPD advantage (sats/$)| +304.9       | +272.9         | −32.0      |

Decision rule: adopt only if RW improves and either (a) win rate does not degrade
materially or (b) mean/absolute SPD advantage improves.

**Amplitude statistics** (vol-normalized variant):
- Mean amplitude: 0.1579 (baseline: 0.1500) — slightly higher on average
- 13.5% of days at the cap (0.22) in low-vol regime; 0% at floor
- Vol ratio (rv_60/rv_ref): mean 1.010, std 0.317; 7.2% high-vol (>1.5), 15.3% low-vol (<0.7)

**Verdict: DO NOT ADOPT (marginal / not worth added complexity)**.
RW improves only +0.14 pp, while win rate declines (−1.59%) and the sats/$ advantage
shrinks (−32). The RW delta is small enough to plausibly reflect scan-level variability
rather than a robust gain, and the deteriorations suggest worse path quality despite
similar terminal RW percentile. The experiment's primary value is diagnostic: holding
ts_z constant and changing only amplitude indicates that most vol-regime adaptation is
already captured by ts_z's standardization, with explicit amplitude scaling largely
redundant.

**Mechanism supported**: ts_z's implicit vol-normalization via the rolling 252d std
denominator is already capturing most of the vol-regime signal. The explicit amplitude
layer adds a largely redundant second normalization:
- High-vol regimes: higher dispersion increases the rolling std in the ts_z denominator
  → ts_z shrinks → tilt already contracts. The explicit rule further contracts amplitude,
  producing a second contraction and reducing SPD edge.
- Low-vol regimes: ts_z tends to enlarge standardized moves; the amplitude cap (0.22)
  adds limited incremental lift (13.5% of days capped), but this does not translate into
  better sats/$ or win rate.

**Narrative value**: Clean experimental design — same ts_z signal, isolated amplitude
layer — supports the interpretation that the z-score's implicit vol-normalization is the
mechanism behind the edge, not any explicit vol scaling.

Script: `vol_amplitude_analysis.py`

---

## 3d) Weekly Granularity Experiment

**Hypothesis**: Resampling daily BTC prices to weekly reduces short-term noise
and better aligns the OLS trend signal with Bitcoin's 6-18 month cycle structure,
potentially improving RW% by 10-20 pp.

**Implementation**: `resample('W-MON').last()` → weekly log prices → `_fast_rolling_ols`
with annualization ×52 (weeks/year) → z-score → forward-fill to daily.

**Parameter sweep** (step=7 coarse scan, ~440 windows, 2016–2025):

| Variant      | short/long/z | RW%   | ΔRW      | Win%   | SPD Adv  |
|--------------|-------------|-------|----------|--------|----------|
| Daily OLS    | 60d/180d/252d | 45.84% | baseline | 66.14% | +304.9   |
| 8/26/52w     | 56d/182d/364d | 44.80% | −1.04 pp | 72.95% | +316.4   |
| 12/26/52w    | 84d/182d/364d | 44.94% | −0.90 pp | 72.50% | +374.5   |
| 12/52/52w    | 84d/364d/364d | 43.44% | −2.40 pp | 71.36% | +312.7   |
| 26/52/52w    | 182d/364d/364d| 42.10% | −3.74 pp | 70.23% | +412.3   |
| **12/26/26w**| 84d/182d/182d | **46.10%** | **+0.26 pp** | 65.00% | +308.6 |

Decision rule: same as all prior experiments — adopt only if RW improves AND
(win rate stable OR SPD advantage improves).

**Verdict: DO NOT ADOPT (null result on weekly granularity)**.

The primary hypothesis is **not supported**: the direct weekly equivalent of
the daily signal (12/26/52w) underperforms daily by −0.90 pp. All configurations
using the "natural" weekly translation of 252d z-score (52w ≈ 1 year) degraded.

The only marginally-positive variant (12/26/26w, +0.26 pp) uses a **shorter
z-score window** (26w ≈ 182d) than the standard 52w equivalent. This suggests
any benefit comes from tuning the z-score window length, not from weekly
resampling per se. The +0.26 pp delta is noise-level and doesn't meet the bar
for adoption.

**Mechanism — why weekly OLS didn't help**:
The daily 252d rolling z-score already filters noise through its std denominator:
- High-vol days → larger rolling std → ts_z shrinks → tilt dampened automatically
- The z-score normalization captures cycle dynamics at daily granularity

Weekly resampling was expected to add further noise reduction, but the z-score
already extracts the relevant signal. Longer weekly windows (26w, 52w) introduce
a larger warm-up period and fewer independent signal updates, which appears to
hurt performance more than the noise reduction helps.

**Narrative value**: This null result confirms that the +3.01 pp edge over neutral
DCA derives from the **z-score normalization structure** (regime-relative dampening),
not from the OLS window size or temporal granularity. The current daily 60/180/252
parameterization appears to be near-optimal for this signal class.

Script: `weekly_granularity_analysis.py`

---

## 4) Quick Demo (Office Hours "30-second proof")

### Primary demo

Run:
```bash
python3 verify_endpoints_demo.py
```

Expected "receipt-like" passes:
- PASS: import + signatures
- PASS: df_out columns preserved + prob_up added
- PASS: weights len=365 sum=1.0 min>=MIN_WEIGHT index aligned

### Plan B fallback (imports + signatures only)

```bash
python3 -c "from tournament_mode import construct_features, compute_weights, MIN_WEIGHT; import inspect; print(construct_features.__name__, compute_weights.__name__); print('MIN_WEIGHT', MIN_WEIGHT); print('construct_features sig:', inspect.signature(construct_features)); print('compute_weights sig:', inspect.signature(compute_weights))"
```

### Setup & smoke tests

```bash
# 1. Install dependencies (pyarrow required for analysis scripts)
pip install -r requirements.txt

# 2. Dependency smoke test (must pass before running analysis scripts)
python -c "import pandas as pd; import pyarrow; print('pandas', pd.__version__, '/ pyarrow OK')"

# 3. Endpoint contract smoke test (must pass before grader demo)
python3 verify_endpoints_demo.py

# 4. Optional: confirm torch backend vs OLS fallback
python -c "from tournament_mode.evaluator import _FEATURES_BACKEND; print('features_backend:', _FEATURES_BACKEND)"
```

Expected output of (4): `features_backend: CNN` (torch available) or `features_backend: SIMPLIFIED` (OLS fallback, grader-safe).

---

## 5) Known Pitfalls + Guardrails

### Template compliance
- Function names must be exact: `construct_features`, `compute_weights`
- Import path must be clean: `from tournament_mode import ...`
- `construct_features` must not drop input columns (price column must remain)

### Date range + calendar gotchas
- Confirm exact tournament date range and inclusivity with sponsor/TA
- Confirm whether weekends exist in official dataset and how windows count days

### "No network" / artifacts
- Assume grader may have restricted network access
- Bundle required artifacts locally if model-based approaches are used
- Prefer minimal deps (pandas/numpy; torch only if explicitly allowed)

---

## 6) Midterm Materials Inventory

### Demo + proof
- `verify_endpoints_demo.py`
- `MIDTERM_ENDPOINTS_PROOF.md`
- `OFFICE_HOURS_RUNBOOK.md`

### Presentation decks
- `AI_Sees_the_Market.pdf` (vision / NotebookLM)
- `VTS_MIDTERM_ENDPOINTS.md` (technical endpoints proof)
- `VTS_MIDTERM_PRESENTATION.md` (results + learning narrative)

### Implementation
- `tournament_mode/__init__.py` (exports)
- `tournament_mode/features_simplified.py` (construct_features)
- `tournament_mode/weights.py` (compute_weights)
- `btc_accumulation_model_simplified.ipynb` (working reference notebook)

---

## 7) Next 3 Tasks (Immediate)

1. **Office hours questions to sponsor/TA (top 8)**:
   - date range inclusivity + weekend handling
   - submission interface (CSV vs callable funcs)
   - official dataset + guaranteed columns (volume)
   - scoring confirmation (rolling windows, rho=0.9, win rate definition)
   - allowed data sources/signals (external data allowed?)
   - grader environment constraints (CPU/runtime/packages)
   - required artifacts + naming + no-network rule
   - common failure modes + guardrails

2. **After office hours**: update this doc with authoritative answers (source of truth)

3. **Decide final-track work**:
   - Document lessons learned (default)
   - Optional performance experiments (weekly horizon / broader training) only if valuable for learning

---

## 8) Decisions Log (short)

- Midterm baseline = neutral `prob_up=0.5` (ablation-based)
- Post-midterm signal = OLS contrarian raw tanh → 43.25% RW (+1.31 pp vs neutral)
- Current signal = OLS contrarian + z-score shrinkage (±2σ clip) → 44.95% RW (+3.01 pp vs neutral)
- Turnover governor (Mode A: freeze_then_ema) implemented; zero RW% impact, ~12% turnover reduction
- Proportional fees are percentile-invariant (proven mathematically, confirmed numerically 0–50 bps)
- Canonical state doc lives at `plans/PROJECT_STATE.md`
- Serena memory used only for code breadcrumbs (paths, commands), not narrative

---

**Post-office hours**: paste the sponsor/TA answers here and I'll revise this doc into a crisp "authoritative spec" section (date range, dataset columns, grader constraints, etc.) so it becomes the single source of truth.
