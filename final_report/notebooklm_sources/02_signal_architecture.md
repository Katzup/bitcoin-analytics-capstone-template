# VTS Technical Deep-Dive: Signal Architecture & Design Decisions

## How the Final Signal Works

The final signal is called **OLS trend-relative z-score dampening**. Here's how it works step by step.

### Step 1: Dual-Window OLS Slope

For each day t, we fit two separate rolling regressions on log-prices — one 60-day window and one 180-day window.

```
slope_60(t) = OLS coefficient on [log P_{t-59}, ..., log P_t]
slope_180(t) = OLS coefficient on [log P_{t-179}, ..., log P_t]
```

Each slope is the best-fit linear trend in log-price — essentially, the estimated annualized growth rate in log-return units.

We weight each slope by its OLS R² (coefficient of determination) to down-weight noisy or flat periods:

```
TrendScore(t) = 0.4 × (slope_60 × R²_60 × 365) + 0.6 × (slope_180 × R²_180 × 365)
```

Multiplying by 365 annualizes the slope from per-day to per-year units.

**Why two windows?** The 60-day window captures recent trend momentum; the 180-day window captures the medium-term regime. The 60/40 weighting gives more influence to the longer window (0.6 weight), which is more stable.

**Why OLS and not raw returns?** OLS fits a line through log-prices, which is more robust to outliers than a simple trailing return. The R² weighting further filters out periods where the trend is too noisy to measure reliably.

### Step 2: Regime-Relative Z-Score Normalization

This is the most important step. We normalize TrendScore relative to its own recent history using a 252-day rolling z-score:

```
ts_z(t) = clip[ (TrendScore(t) − μ_{252}(t)) / σ_{252}(t),  −2, +2 ]
```

Where μ and σ are the 252-day rolling mean and standard deviation of TrendScore (not of raw price).

The clip at ±2σ is called **winsorization** — it prevents extreme signals from dominating allocation.

**Why z-score normalize?** This is regime-relative normalization. The same absolute trend slope (e.g., +50% annualized log-return trend) is treated differently depending on recent market context:
- If recent TrendScore volatility (σ_{252}) is high (volatile regime), the same slope produces a smaller ts_z → smaller tilt
- If recent TrendScore volatility is low (quiet regime), the same slope produces a larger ts_z → larger tilt

This is why the signal is called "regime-relative": the z-score adapts the signal's strength to the current market environment without explicit regime detection.

**Implicit vol-normalization:** The 252-day rolling std in the denominator already normalizes for realized volatility regimes. High-volatility periods (where BTC trends are harder to sustain) naturally reduce the standardized signal magnitude. An additional explicit vol-amplitude layer was tested and rejected — it was redundant.

### Step 3: Nonlinear Mapping to Probability

```
prob_up(t) = 0.5 − 0.15 × tanh(ts_z(t))
```

Key features of this mapping:
- **Bounded:** prob_up ∈ [0.35, 0.65] due to tanh saturation. The strategy never tilts more than 30% away from neutral.
- **Smooth:** tanh is differentiable, producing continuous rather than binary allocation changes.
- **Inverted (minus sign):** Higher ts_z → lower prob_up → less investment that day. This is not classical "contrarian" behavior — diagnostics show ts_z is *positively correlated* with future returns (r ≈ +0.14 at 30 days). The signal is better described as **dampening extended uptrends** rather than "buying dips."

**Signal interpretation:** When ts_z is high (recent trend is strong relative to its own history), the signal says "we're in a stretched regime, invest slightly less than neutral." When ts_z is low (trend is weak or negative relative to history), the signal says "invest slightly more than neutral." This produces a small, persistent sats-per-dollar advantage over uniform DCA without requiring large bets or precise price prediction.

---

## Turnover Governor

The final signal includes a **turnover governor** that reduces day-to-day weight churn:

- If the day-over-day absolute change in target weight is < 2% (max-delta threshold), carry forward the previous weight (freeze).
- Otherwise apply EMA smoothing: new_weight = 0.30 × target + 0.70 × previous.

The governor froze ~62% of days on real BTC data (2020–2022), reducing turnover by ~12% with **zero impact on RW percentile** (45.84% with and without governor). The existing EMA (α = 0.30) provides most within-window damping; the governor adds operational stability at negligible cost.

---

## From Signal to Tournament Submission

The signal produces a `prob_up` value for each day. The tournament framework converts this to portfolio weights:

1. `prob_up` is passed to `compute_weights()`, which applies EMA smoothing and the turnover governor
2. Weights are clipped to ensure each day's weight ≥ MIN_WEIGHT (no day gets zero allocation)
3. Weights are normalized to sum to 1 within each rolling evaluation window
4. The SPD (sats per dollar) is computed using these weights × (1e8 / price)

The tournament evaluates the *relative rank* of this SPD within each window — not the absolute level — so proportional transaction fees cancel out entirely.

---

## Design Choices Rejected After Testing

The following variants were explicitly tested and rejected:

### Vol-Normalized Amplitude (+0.14 pp RW, REJECTED)
**Hypothesis:** Scale the tilt amplitude inversely with realized volatility (rv_60/rv_ref). In high-volatility regimes, reduce the amplitude of prob_up deviations from 0.5.

**Result:** +0.14 pp RW improvement but win rate declined (−1.59%) and absolute SPD advantage shrank (−32 sats/$). The ts_z denominator already adapts to volatility — the explicit layer is redundant and degrades path quality.

### Weekly OLS Resampling (−0.90 pp RW, REJECTED)
**Hypothesis:** Weekly prices reduce daily noise and align better with BTC's 6-18 month market cycles.

**Result:** The direct weekly equivalent (12/26/52 weeks) underperformed by −0.90 pp. The best weekly variant (12/26/26w) showed marginal +0.26 pp gain, but this comes from a shorter z-score window (26w ≈ 182 days), not from weekly resampling itself.

**Conclusion:** The daily 60/180/252-day design is near-optimal for this signal class.

---

## Architecture in Code

The final signal is implemented in `tournament_mode/features_simplified.py`:

```python
def build_features_trend_ols(df):
    """Rolling dual-window OLS trend score with z-score normalization."""
    # Step 1: OLS slopes (60d, 180d) weighted by R²
    for w in [60, 180]:
        slope, r2 = rolling_ols_slope(log_prices, window=w)
        df[f'slope_{w}'] = slope * 365  # annualize
        df[f'r2_{w}'] = r2

    df['TrendScore'] = (0.4 * df['slope_60'] * df['r2_60'] +
                        0.6 * df['slope_180'] * df['r2_180'])

    # Step 2: 252-day z-score normalization, clipped at ±2σ
    mu = df['TrendScore'].rolling(252).mean()
    sigma = df['TrendScore'].rolling(252).std()
    df['ts_z'] = ((df['TrendScore'] - mu) / sigma).clip(-2, 2)

    # Step 3: Nonlinear mapping to probability
    df['prob_up'] = 0.5 - 0.15 * np.tanh(df['ts_z'])

    return df
```

The key design invariant: `prob_up` ∈ [0.35, 0.65] — bounded, smooth, causal, interpretable.
