# VTS Results, Ablation Studies & Null Results

## Primary Results: The Performance Ladder

The tournament uses Recency-Weighted (RW) Percentile as the primary metric. Here is the full performance ladder from weakest to strongest:

| Signal | RW% | Win% | vs Neutral |
|--------|-----|------|-----------|
| CNN/GAF (image classification) | 41.43% | 54.32% | −0.51 pp |
| **Neutral DCA (prob_up = 0.5)** | **41.94%** | **70.42%** | **baseline** |
| OLS raw contrarian (no z-score) | 43.25% | 91.48% | +1.31 pp |
| **OLS z-score winsorized ±2σ** | **44.95%** | **66.94%** | **+3.01 pp** |

All evaluations: step=1, 3,076 rolling 365-day windows, 2016-01-01 through 2025-06-01, decay=0.9, seed=42.

## Interpreting the +3.01 pp Headline

The 44.95% RW vs. 41.94% neutral means:
- **+3.01 percentage points** improvement in sats-per-dollar timing quality
- This advantage is **persistent**, not episodic: the strategy outperforms neutral DCA in **66.94%** of the 3,076 evaluation windows (2,060 of 3,076 windows)
- The advantage appears throughout the 2016–2025 period, not concentrated in one market regime

**What the numbers mean physically:** In a 365-day window where BTC's price ranges from $30,000 to $60,000 (SPD_min = 1,667 sats/$, SPD_max = 3,333 sats/$), a 3 pp improvement in percentile means accumulating approximately 50 more sats per dollar than neutral DCA — roughly equivalent to buying at a price that is ~$1,500 lower than the neutral strategy's average purchase price.

## Why the CNN Failed: Root Cause Analysis

The CNN/GAF approach achieved 41.43% RW — *below* the 41.94% neutral baseline by 0.51 pp. This is a clear failure.

**Root cause:** Pattern recognition accuracy ≠ trading edge. The CNN was optimized to classify candlestick patterns with high F1 score on held-out labels. But:
1. The labels (bullish/bearish patterns) were defined by historical chart analysis, not by tournament SPD outcomes.
2. Even if patterns were perfectly classified, the *direction* they predict may not align with the tournament's accumulation metric.
3. The CNN was trained on 2014–2015 data (pre-institutional era) and deployed on 2016–2025 data (very different BTC regime).

**The key insight:** You can have a 90% accurate image classifier that provides zero exploitable edge in a specific trading metric. The classification task and the trading task are different objective functions.

## The Pivot: Why OLS?

After the CNN failure, we chose OLS for three reasons:
1. **Causality:** OLS regressions on past prices only — no look-ahead
2. **Interpretability:** The slope coefficient has a clear meaning (trend strength)
3. **Diagnosability:** We can isolate each design choice (raw slope vs. z-scored, 60d vs. 180d windows, raw vs. winsorized) and measure its contribution

**Raw OLS to z-scored OLS:** The raw contrarian (no z-score) achieved 43.25% RW with 91.48% win rate. Adding the 252-day z-score normalization improved RW to 44.95% (+1.70 pp) but reduced win rate to 66.94%. The z-score makes the signal regime-relative: instead of asking "is the trend strong?" it asks "is the trend strong *relative to recent history?*" This produces a more persistent and meaningful signal.

## Ablation Study 1: Vol-Normalized Amplitude

**Hypothesis:** Scale the tilt amplitude inversely with short-horizon realized volatility. In high-vol regimes, reduce the tilt's impact.

**Implementation:**
```
vol_norm_factor = clip(rv_60 / rv_ref, 0.5, 2.0)
prob_up = 0.5 - 0.15 × (1/vol_norm_factor) × tanh(ts_z)
```

**Result:**
- RW: +0.14 pp (45.99% vs. 45.84%, fast-scan)
- Win rate: −1.59%
- Mean excess SPD: −32 sats/$

**Verdict:** REJECTED. The marginal RW gain (+0.14 pp) is within scan-level noise, and the deterioration in win rate and SPD advantage indicates worse path quality. The ts_z z-score already adapts to volatility regimes through its 252-day rolling std denominator. Adding an explicit short-horizon layer is redundant and harmful.

**Causal story:** In high-vol regimes → rolling std (σ_{252}) is already elevated → ts_z is already smaller → tilt is already contracted. Adding an explicit amplitude reduction on top creates *double contraction*, which reduces the SPD advantage without improving the distribution of purchase prices.

## Ablation Study 2: Weekly OLS Resampling

**Hypothesis:** Weekly prices reduce daily noise and align with BTC's 6–18 month market cycles. A weekly OLS (12/26/52-week windows) should outperform the daily version.

**Key configurations tested:**

| Variant | RW% (fast-scan) | ΔRW vs daily | Notes |
|---------|-----------------|--------------|-------|
| Daily baseline (60/180/252d) | 45.84% | — | Current best |
| Weekly 12/26/52w (primary) | 44.94% | −0.90 pp | Direct weekly equiv |
| Weekly 12/26/26w | 46.10% | +0.26 pp | Shorter z-window |
| Weekly 4/12/12w | 44.35% | −1.49 pp | Short-window variant |
| Weekly 12/52/52w | 43.26% | −2.58 pp | Long-window variant |
| Weekly 8/26/52w | 44.04% | −1.80 pp | Hybrid |
| Weekly 26/52/104w | 42.10% | −3.74 pp | Very long window |

**Key finding:** The best weekly variant (12/26/26w, +0.26 pp) achieves its gain from a *shorter z-score window* (26w ≈ 182d vs. 52w ≈ 364d), not from weekly resampling itself. The direct weekly equivalent (12/26/52w, the theoretically correct analog) underperformed by −0.90 pp.

**Verdict:** REJECTED. Weekly granularity does not improve the signal. The daily 60/180/252-day design is near-optimal for this signal class. The gain from shorter z-windows is already captured by the daily design's 252-day (≈ 52-week) z-score window.

## Turnover Governor Ablation

**Component:** freeze-then-EMA governor added to `compute_weights()`

**Effect:** Froze ~62% of days (real BTC data 2020–2022), reducing turnover by ~12% vs. EMA-only.

**RW impact:** Zero (45.84% with and without governor).

**Verdict:** ADOPTED. Provides operational stability at negligible cost. The existing EMA (α = 0.30) does the heavy lifting for within-window smoothing; the governor reduces day-to-day churn for no RW cost.

## Fee Robustness

The RW percentile is **mathematically invariant to proportional transaction fees** in this evaluation harness. The proof:

Under proportional fees (fee factor f = 1 + fee_bps/10,000):
```
SPD_strategy_fee = SPD_strategy / f
SPD_min_fee = SPD_min / f
SPD_max_fee = SPD_max / f
```

Therefore:
```
pct_fee = (SPD_strategy/f − SPD_min/f) / (SPD_max/f − SPD_min/f) = pct_nofee
```

The f cancels, and the percentile is unchanged. This invariance holds because fees scale all SPD values within a window uniformly.

**Breakeven fee (absolute):** ~40–45 bps, where the absolute SPD advantage (sats per dollar edge) approaches zero. Above this fee level, the strategy still has the same RW percentile rank but the absolute sats-per-dollar edge is erased by fee drag.

## Traditional Metrics (Supplementary)

To cross-check the RW result against familiar risk/return intuition, we computed equity curve metrics for a $1/day DCA simulation (Aug 2017 – Jun 2025, ~8.3 years):

| Metric | Strategy | Neutral DCA | Δ |
|--------|----------|-------------|---|
| CAGR | 42.6% | 38.5% | +4.1 pp |
| Ann. volatility | 58.1% | 58.1% | ≈ 0 |
| Sharpe ratio | 0.611 | 0.561 | +0.050 |
| Sortino ratio | 0.772 | 0.709 | +0.063 |
| Max drawdown | −79.0% | −81.6% | +2.5 pp better |
| Calmar ratio | 0.539 | 0.472 | +0.067 |
| BTC accumulated | 0.695× neutral | — | −30.5% |

**CAGR caveat:** The 365-day warm-up period (2016–Aug 2017) spans a major BTC bull run. The dampening signal was already active, producing a lower initial portfolio value for the strategy at simulation start. CAGR is confounded by different starting conditions. Sharpe, Sortino, drawdown, and Calmar are the reliable comparators.

**BTC accumulation:** The strategy accumulated ~31% fewer BTC than neutral over 2017–2025. This is expected: in a secular bull market (the dominant regime 2017–2025), regime-relative dampening chronically reduces investment below the rolling baseline during extended uptrends. The strategy's edge is window-based timing quality (better sats/$ within each 365-day window), not maximum BTC accumulation over a full supercycle.

**Rolling MD/RV:** The strategy had lower max-drawdown / realized-volatility ratio in **60.7% of quarterly windows** (mean 0.83 vs 0.84 for neutral). The ratio exceeded 1.0 only during the 2018–19 and 2022 bear markets (peak 1.30 for strategy vs 1.31 for neutral). This confirms the strategy does not amplify drawdowns.

**Conclusion:** The traditional metrics confirm that the +3.01 pp timing edge does not degrade risk-adjusted returns. Sharpe (+0.05), Sortino (+0.06), max drawdown (−2.5 pp better), and rolling MD/RV (lower in 61% of windows) all improve modestly.
