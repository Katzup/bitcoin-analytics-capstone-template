# Can Timing Beat Bitcoin DCA? What a Georgia Tech Practicum Taught Me

*A deliberately simple signal outperformed constant dollar-cost averaging — not by predicting price, but by reading the volatility regime.*

---

Most Bitcoin accumulation advice boils down to: buy the same amount every day and don't look at the chart. Dollar-cost averaging (DCA) is simple, disciplined, and hard to beat. So when Georgia Tech's OMSA program challenged me to try, I expected the answer to be "no."

I was almost right.

## The Hypothesis That Failed

The original idea was straightforward: train a CNN on candlestick chart images to classify next-day price direction, then tilt daily Bitcoin purchases based on the model's confidence. If the model says "up," buy more. If "down," buy less.

The CNN achieved respectable classification accuracy during training. But when fed through the tournament's actual evaluation metric — recency-weighted sats-per-dollar across 3,076 rolling windows from 2016–2025 — it scored 41.43%. A neutral baseline that simply buys the same amount every day? 41.94%. The CNN was indistinguishable from a coin flip in the metric that mattered.

Pattern recognition is not the same as a trading edge.

## What Actually Worked

Rather than abandon the project, I stripped the approach down to something almost embarrassingly simple.

Take the OLS trend slope of Bitcoin's log price over a 60-day window. Normalize it as a z-score against a 252-day rolling standard deviation. Clip outliers at ±2 standard deviations. Then convert to a daily purchase weight:

**prob_up = 0.5 − 0.15 × tanh(z-score)**

That's the entire signal. When Bitcoin's trend is unusually strong relative to recent volatility, the formula slightly reduces your daily purchase. When the trend is unusually weak, it slightly increases it. The tilt is modest — never more than ±15 percentage points from equal weighting.

This scored **44.95% on the tournament percentile** — a +3.01 percentage point improvement over neutral DCA.

Three points doesn't sound like much. But in a tournament where thousands of rolling windows are evaluated and most signals cluster near the baseline, a consistent 3-point edge is meaningful. It means you're accumulating more sats per dollar across a wide variety of market conditions.

*Figure 1: Strategy comparison across 3,076 rolling evaluation windows (2016–2025). Left: Tournament metric (RW-SPD percentile) — the OLS z-score signal achieves 44.95%, a +3.01pp improvement over neutral DCA (41.94%). The CNN scored 41.43%, worse than doing nothing. Right: Win rate vs naive DCA across all windows. Data from Trilemma Foundation Stacking Sats Tournament.*

## Why It Works (and What It Isn't)

The natural assumption is "buy the dip" — the signal increases purchases when prices drop. But the data tells a more nuanced story.

The z-score is positively correlated with forward 30-day returns (r ≈ +0.14). That's momentum-like, not contrarian. So why does inverting it help accumulation?

The answer is **regime-relative dampening**. By normalizing the trend against a 252-day rolling standard deviation, the signal automatically adapts to the current volatility regime. In calm markets, small moves register as meaningful z-scores. In volatile markets, even large swings get dampened. The signal doesn't predict direction — it recognizes when the market is behaving unusually relative to its own recent history, and adjusts purchase sizes accordingly.

Formal statistical tests support the regime structure. Ljung-Box testing confirmed significant volatility clustering (Q = 847.3, p < 0.001). Mann-Whitney U tests showed bull and bear regime return distributions are statistically distinct (p < 0.001). Bootstrap confidence intervals put mean daily returns at +0.23% in bull regimes versus −0.09% in bear regimes.

The signal exploits these regime differences — not by timing tops and bottoms, but by spending slightly less during overextended conditions and slightly more during depressed ones.

*Figure 2: Formal regime evidence panel. Left: Ljung-Box Q-statistic (Q = 847.3, p < 0.001) confirming significant volatility clustering in Bitcoin daily returns. Center: Mann-Whitney U test showing bull and bear regime return distributions are statistically distinct (p < 0.001). Right: Bootstrap 95% confidence intervals — mean daily returns of +0.23% in bull regimes versus −0.09% in bear regimes. These structural regime differences are what the OLS z-score signal is designed to exploit.*

## What I Tried That Didn't Work

Science is honest about null results. Several ideas that seemed promising added no value:

**Volatility-amplitude scaling** — adjusting the signal strength based on short-term realized volatility. Result: +0.14 percentage points on the tournament metric but worse win rate (−1.6%) and lower absolute sats-per-dollar advantage. The z-score already captures volatility implicitly through its rolling standard deviation. Adding an explicit layer was redundant.

**Weekly granularity** — resampling to weekly signals to reduce daily noise. The direct weekly equivalent underperformed by −0.90 percentage points. The daily signal's edge comes from its normalization structure, not from the frequency of updates.

**Fee sensitivity** — I proved mathematically that proportional transaction fees are percentile-invariant in this evaluation framework. The fee factor cancels in the percentile ratio. The strategy's breakeven fee is approximately 40–45 basis points, well above typical exchange costs.

A **turnover governor** was adopted — not for performance (it had zero impact on the tournament metric) but for operational stability, reducing portfolio turnover by roughly 12%.

## What This Means for Bitcoin Accumulators

If you're dollar-cost averaging into Bitcoin, you're already doing something sensible. This research suggests you can do slightly better by paying attention to one thing: how the current price trend compares to recent volatility.

When Bitcoin has been trending strongly relative to its own recent behavior, buy a little less. When the trend is weak relative to recent volatility, buy a little more. Not dramatically — just a modest tilt.

The key insight isn't about prediction. It's about **regime awareness**. Bitcoin's volatility clusters. Bull and bear periods have statistically different return distributions. A simple normalization that respects these regimes captures most of the available edge. More complex approaches — CNNs, volatility overlays, weekly resampling — don't add meaningful value over this baseline.

Sometimes the best model is the simplest one that accounts for structure in the data.

## Explore the Full Analysis

The complete codebase, data pipeline, and reproducibility package are available on GitHub:

**[GitHub Repository](https://github.com/Katzup/bitcoin-analytics-capstone-template)**

The repo includes a 30-second smoke test (`make all`), formal statistical tests, and every experiment described above with full results. A 7-minute video walkthrough covers the project for non-technical audiences:

**[Bitcoin - Win by Failing First](https://youtu.be/dPQTMGjO_cM)**

*This work was completed as part of Georgia Tech's OMSA Practicum (Spring 2026) in partnership with the [Trilemma Foundation](https://www.stackingsats.org/). The analysis covers Bitcoin accumulation strategies from 2016–2025 using publicly available price data.*

---

*Bob Katz is the founder of FACTS Consulting, where he works at the intersection of Finance, Analytics, Consulting, Transformation, and AI. Connect on [LinkedIn](https://linkedin.com/company/facts-consulting) or visit [factservices.com](https://factservices.com).*
