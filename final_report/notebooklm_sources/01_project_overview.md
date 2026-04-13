# VTS Practicum: Project Overview — Trilemma Stacking-SATS Tournament
# GT OMSA Practicum | Bob Katz | Spring 2026

## What Is This Project?

This is a Georgia Tech OMSA (Online Master of Science Analytics) practicum project. The goal was to design and evaluate a Bitcoin Dollar-Cost Averaging (DCA) strategy that beats a neutral baseline in the Trilemma Foundation's "Stacking-SATS" tournament.

**The tournament question:** Can a data-driven signal systematically improve the timing of BTC purchases to accumulate more sats (Bitcoin satoshis) per dollar spent, compared to a completely uniform DCA baseline?

A "sat" is 1/100,000,000 of a Bitcoin. If BTC costs $50,000, each dollar buys 2,000 sats. If BTC costs $40,000, each dollar buys 2,500 sats. The tournament rewards strategies that time purchases to get more sats per dollar — not by making huge bets, but by consistently tilting purchases slightly toward lower prices.

---

## The Tournament Setup

**Data:** Bitcoin price from 2016-01-01 through 2025-06-01 (Coinmetrics data, ~3,400 days)

**Evaluation:** Rolling 365-day windows, stepped by 1 day → 3,076 evaluation windows

**Primary metric:** Recency-Weighted (RW) Percentile of sats-per-dollar (SPD) rank within each 365-day window. Newer windows count more (exponential decay = 0.9). Higher is better.

**SPD formula:** Within each 365-day window, SPD = sum of (weight × sats_per_dollar_on_day_t), where weights reflect the strategy's allocation. The neutral DCA baseline uses equal weights every day.

**Percentile scoring:** For each window, the strategy's SPD is converted to a percentile:
- 0% = bought everything at the window's peak price (worst possible)
- 100% = bought everything at the window's trough price (best possible)
- 50% = strategy split exactly between min and max prices

**Neutral baseline:** Constant prob_up = 0.5, uniform allocation. Achieves **41.94% RW percentile, 70.42% win rate** over the full evaluation period.

**Win rate:** Secondary metric — fraction of 365-day windows where the strategy's SPD percentile exceeds neutral DCA's percentile.

---

## The Journey: Three Phases

### Phase 1 — CNN/GAF Approach (Pre-Midterm)

**The idea:** Use image classification. Convert BTC price data into Gramian Angular Fields (GAF) — a 2D image representation of a time-series — and train a Convolutional Neural Network (CNN) to classify candlestick patterns. If the CNN says "this pattern leads to price drops," reduce investment that day.

**The logic:** CNNs have shown high classification accuracy on candlestick patterns in academic literature. If patterns are predictive, a CNN should capture them.

**The execution:** Trained on 2014–2015 historical data, evaluated on 2016–2025 tournament period.

**The result:** CNN/GAF achieved **41.43% RW percentile** — *below the neutral DCA baseline* (41.94%). The approach didn't just fail to help; it slightly hurt performance.

**Why it failed (root cause):** There is a fundamental distinction between *classification accuracy* and *trading edge*. A CNN can learn to identify candlestick patterns with high F1 score while those patterns provide zero exploitable predictability for accumulation timing. Image classification accuracy does not transfer to tournament-relevant behavior under the sats-per-dollar metric.

### Phase 2 — The Pivot (Post-Midterm)

After the CNN failure, the project pivoted away from deep learning toward a transparent, interpretable signal. The pivot criteria were:
1. **Causal** — only uses past price data, no look-ahead
2. **Interpretable** — each component has a documented economic rationale
3. **Diagnosable** — ablations can isolate the contribution of each design choice

### Phase 3 — OLS Signal Engineering (Final Signal)

Built an Ordinary Least Squares (OLS) trend-relative dampening signal. The final signal achieves **44.95% RW percentile (+3.01 pp over neutral DCA)**.

---

## Why This Matters

The project demonstrates several important principles:

1. **Sophisticated ML ≠ better performance.** A simple linear OLS signal outperformed a deep learning CNN by +3.52 pp. Interpretability and diagnosability matter.

2. **Null results are findings.** Two high-potential variants (vol-normalized amplitude, weekly OLS resampling) were tested and rejected. These null results strengthen the methodology by showing the +3.01 pp edge is specific to the z-score structure, not arbitrary.

3. **Scientific integrity.** The strategy accumulates fewer BTC overall in a secular bull market (−31%) — this is documented honestly, not hidden. The strategy's edge is *timing quality* within windows, not maximum accumulation.

4. **Tournament invariance.** Proportional transaction fees don't affect the percentile ranking — mathematically proven. The evaluation harness cancels fee effects.

---

## Key Numbers (Locked)

| Signal | RW% | Win% | vs Neutral |
|--------|-----|------|-----------|
| Neutral baseline (prob_up = 0.5) | 41.94% | 70.42% | — |
| CNN/GAF | 41.43% | 54.32% | −0.51 pp |
| OLS raw contrarian | 43.25% | 91.48% | +1.31 pp |
| **OLS z-score winsorized ±2σ** | **44.95%** | **66.94%** | **+3.01 pp** |

Evaluation: step=1, 3,076 rolling 365-day windows, 2016-01-01 to 2025-06-01, decay=0.9, seed=42.
