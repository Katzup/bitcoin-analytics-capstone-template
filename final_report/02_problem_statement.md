# Section 2: Problem Statement & Motivation
# STATUS: DRAFT

---

## 2.1 The Bitcoin Accumulation Problem

Bitcoin's fixed supply schedule and high price volatility create a structurally interesting accumulation problem. Unlike equities, where dividends and earnings provide fundamental valuation anchors, Bitcoin's near-term price is dominated by momentum, sentiment, and macro liquidity cycles. This makes the asset simultaneously difficult to value and potentially amenable to systematic timing — if any timing edge exists at all.

The practical accumulation question is simple: given a fixed capital commitment over a one-year horizon, does *when* you buy matter? A uniform daily dollar-cost averaging (DCA) strategy eliminates timing decisions entirely by spreading purchases equally across all trading days. Any signal-based strategy must justify its complexity by outperforming this passive benchmark on a risk-adjusted basis.

## 2.2 Tournament Framing and Metric Definition

This project operates within a Bitcoin Accumulation Tournament evaluated using a recency-weighted sats-per-dollar (RW-SPD) metric. The metric rewards strategies that accumulate more satoshis per dollar spent, with recent purchases weighted more heavily than historical ones.

**Sats-per-dollar for a single purchase** on day *t* at price *P_t*:

$$\text{SPD}_t = \frac{10^8}{P_t}$$

**Strategy SPD** for a 365-day window with daily allocation weights {*w_t*}:

$$\text{SPD}_\text{strategy} = \sum_{t=1}^{365} w_t \cdot \frac{10^8}{P_t}$$

where weights *w_t* ≥ 0 and sum to 1. The allocator's job is to tilt weights toward days where Bitcoin is cheaper relative to the window's price distribution.

**Recency-weighted percentile rank** across *N* = 3,076 rolling 365-day windows (2016–2025, step=1, decay=0.9):

$$\text{RW percentile} = \frac{\sum_i \lambda^{N-i} \cdot \mathbf{1}[\text{SPD}_i^\text{strat} > \text{SPD}_i^\text{median}]}{\sum_i \lambda^{N-i}} \times 100$$

where λ = 0.9 is the recency decay factor and the median is computed across all submitted strategies in window *i*. A score of 50% corresponds to exactly median performance; the neutral DCA baseline achieves **41.94% RW** in our single-strategy evaluation, since constant weights systematically underperform the median of a distribution that includes strategies that succeeded in hindsight.

**The neutral baseline** is operationalized as constant `prob_up = 0.5` passed through the same smoothed-EMA allocator. This produces uniform daily weights and serves as the practical lower bound for any signal-based strategy.

## 2.3 Why Timing Might Work — and Why It Is Hard

Bitcoin exhibits well-documented momentum and mean-reversion regimes at different time scales: trend-following works over multi-month horizons; short-term reversion is more common at daily and weekly scales. The existence of these patterns suggests that regime-relative signal engineering — rather than raw price prediction — may offer a tractable timing edge.

However, the tournament metric introduces two important complications:

1. **Within-window rank sensitivity**: SPD percentile is measured *within* each 365-day window. A strategy that buys slightly less on peak-price days achieves the same RW improvement as one that correctly predicts large drawdowns, as long as it does so *consistently* across windows. This rewards stability over speculative precision.

2. **Win-rate / magnitude trade-off**: The recency-weighted percentile rewards *winning windows* rather than magnitude of outperformance. A strategy that beats the median in 70% of windows at +0.5 SPD/window outperforms one that wins 40% at +5.0 SPD/window, all else equal. This penalizes aggressive concentrated timing bets.

These properties motivate our final design philosophy: small, persistent tilts away from peak-priced periods — operationalized through regime-relative z-score normalization — rather than large concentrated bets on predicted drawdowns.

## 2.4 Project Scope and Contributions

This project makes two primary contributions:

1. **Performance contribution**: A signal-engineered OLS z-score strategy achieves **44.95% RW percentile** (+3.01 pp over neutral DCA) evaluated across 3,076 rolling windows, outperforming the neutral baseline in **66.94%** of evaluation windows.

2. **Methodological contribution**: Two high-potential signal variants — volatility-normalized amplitude scaling and weekly OLS resampling — were tested against explicit acceptance criteria and rejected. The negative results narrow the explanation of the timing edge to the z-score normalization structure specifically, and strengthen confidence that the observed advantage is not an artifact of design flexibility.

---

*Section 3 describes the midterm CNN/GAF approach and its failure mode. Section 4 describes the post-midterm pivot to OLS-based signal engineering.*
