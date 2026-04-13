# Section 4: Methods — Post-Midterm Signal Design
# STATUS: DRAFT (invariant — content will not change with sponsor formatting)

---

## 4.1 Strategic Pivot: From Pattern Recognition to Signal Engineering

The midterm evaluation revealed that the CNN/GAF approach performed below the neutral DCA baseline
(41.43% RW vs 41.94% neutral). The root cause: the CNN was trained to recover candlestick pattern
labels (a classification task), but classification accuracy does not imply that identified patterns
produce exploitable accumulation timing under the tournament's sats-per-dollar metric — a distinction
consistent with the broader literature on pattern-recognition vs. trading-edge separation.
Rather than tuning CNN hyperparameters, the project pivoted to a transparent, diagnosable signal
family based on ordinary least-squares trend estimation.

The pivot criteria were: (1) causal — uses only past price data, (2) interpretable — each component
has a documented economic rationale, and (3) diagnosable — ablations can isolate the contribution
of each design choice.

## 4.2 Signal Construction: OLS Trend-Relative Dampening

The final signal uses a two-window rolling OLS on log-prices, combined with z-score normalization
and a bounded nonlinear mapping to a probability tilt.

**Step 1 — Rolling OLS slope estimation**

For each day t, two rolling windows (60-day and 180-day) estimate the log-price trend slope
via ordinary least squares:

    slope_w(t) = OLS coefficient on [log P_{t-w+1}, ..., log P_t]

The slope is weighted by the OLS R² to down-weight noisy periods:

    TrendScore(t) = 0.4 × (slope_60 × R²_60 × 365) + 0.6 × (slope_180 × R²_180 × 365)

The OLS slope is in log-return units per day (Δlog P per day); multiplying by 365 annualizes it
to an approximate annualized growth rate in log-return units.

**Step 2 — Regime-relative z-score normalization**

A 252-day rolling z-score normalizes TrendScore relative to its own recent history:

    ts_z(t) = clip[ (TrendScore(t) − μ_{252}(t)) / σ_{252}(t),  −2, +2 ]

where μ and σ are the rolling mean and standard deviation of TrendScore itself
(not of raw price). The clip at ±2σ (winsorization) prevents extreme signals from
dominating allocation. Critically, this normalization is **regime-relative**: the same
absolute trend level produces a smaller ts_z when recent TrendScore volatility (σ_{252})
is elevated — implicitly dampening the tilt in high-volatility market regimes.

**Step 3 — Nonlinear mapping to probability tilt**

    prob_up(t) = 0.5 − 0.15 × tanh(ts_z(t))

The inversion (minus sign) reduces allocation slightly during extended uptrends
(high ts_z), consistent with a systematic reduction of exposure when the market
is historically stretched relative to recent conditions.
The tanh function provides smooth, bounded behavior: prob_up ∈ [0.35, 0.65].

**Signal interpretation**

Diagnostic analysis (see `contrarian_signal_diagnostics.py`) confirmed that ts_z is
**positively correlated** with forward returns at all tested horizons (r ≈ +0.14 at 30 days).
The signal is therefore better described as **regime-relative dampening** — it accumulates
slightly less during periods of sustained, high-momentum trends — rather than classical
contrarian ("buy the dip") behavior. The positive correlation is not a defect: the strategy
does not need to predict price reversals. It accumulates at a rate that is modestly less than
neutral during extended uptrends and modestly more during subdued periods, producing a small
persistent SPD advantage over uniform DCA rather than a large but intermittent one.

## 4.3 Turnover Governor

To reduce day-to-day weight churn without affecting long-run performance, a freeze-then-EMA
governor was implemented in `compute_weights()`. At each day t:

- If the element-wise absolute change |target_weight[t] − smoothed_weight[t-1]| < 0.02
  (max-delta threshold), carry forward the previous weight (freeze).
- Otherwise apply EMA smoothing: w̃[t] = 0.30 × target[t] + 0.70 × w̃[t-1].

On real BTC data (2020–2022), the governor froze ~62% of days, reducing turnover by ~12%
versus vectorized EMA alone, with zero impact on RW percentile.

**Attribution**: The existing EMA (α = 0.30) provides most within-window damping.
The governor adds operational stability at negligible cost.

## 4.4 Design Choices Rejected After Testing

The following variants were tested and rejected on the basis of the primary metric
(RW percentile) and secondary diagnostics (win rate, absolute SPD advantage):

| Variant | ΔRW | Verdict |
|---------|-----|---------|
| Vol-normalized amplitude | +0.14 pp | REJECTED — win rate ↓, SPD ↓; ts_z normalization already captures vol regime |
| Weekly OLS resampling (12/26/52w) | −0.90 pp | REJECTED — granularity does not help; daily 60/180/252 design is near-optimal |
| Gov. parameter variations | ~0 pp | Confirmed robustness; default params adequate |

---

*See `plans/PROJECT_STATE.md` for full experiment logs, numerical results, and decision rationale.*
