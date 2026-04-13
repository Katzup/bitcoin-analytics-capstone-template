# Section 1: Executive Summary
# STATUS: DRAFT (invariant — content will not change with sponsor formatting)

---

This project investigates whether signal-engineered Bitcoin accumulation strategies can achieve
a persistent timing advantage over uniform dollar-cost averaging (DCA) under the tournament's
recency-weighted sats-per-dollar metric. The project followed a two-phase trajectory: an initial
convolutional neural network (CNN) approach built at the midterm, and a subsequent pivot to
transparent OLS-based signal engineering after the CNN failed to outperform the neutral baseline.

**Midterm result**: The CNN approach, which encoded 30-day BTC candlestick charts as
Gramian Angular Fields (GAF) and classified them by predicted return direction, achieved
**41.43% RW percentile** — 0.51 percentage points *below* the neutral DCA baseline (41.94%).
This negative result, while disappointing, was informative: classification accuracy on historical
candlestick patterns does not imply exploitable predictability under the tournament's
sats-per-dollar metric, which penalizes buying near window peaks regardless of pattern frequency.

**Post-midterm pivot**: The project pivoted to rolling Ordinary Least Squares (OLS) trend
estimation combined with regime-relative z-score normalization. The final signal computes a
two-window (60/180-day) OLS slope on log-prices, normalizes it against a 252-day rolling
baseline (producing a standardized z-score clipped at ±2σ), and maps the result to a bounded
probability tilt via `prob_up = 0.5 − 0.15 × tanh(ts_z)`. The inversion reduces accumulation
slightly when the market is historically stretched relative to recent conditions.

**Final result**: The OLS z-score winsorized signal achieves **44.95% RW percentile**,
representing a **+3.01 percentage point improvement over neutral DCA** evaluated across
3,076 rolling 365-day windows (2016–2025, step=1, decay=0.9). The strategy outperforms
neutral DCA in 66.94% of windows, with the advantage present throughout the full evaluation
period rather than confined to a single market regime. Traditional risk metrics (Sharpe +0.050,
max drawdown −2.5 pp improvement) corroborate the primary result.

The project's secondary contribution is methodological discipline: two high-potential variants
— volatility-normalized amplitude scaling and weekly OLS resampling — were tested and rejected
on the basis of rigorous ablation criteria, confirming that the timing edge derives specifically
from the z-score normalization structure, not from arbitrary design choices.

---

*Sections 4–5 detail the signal construction and numerical results. Section 6 covers ablation studies.
Section 7 covers governance and reproducibility. Section 8 concludes with limitations and future directions.*
