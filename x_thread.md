# X Thread — Bitcoin DCA Signal (VTS Practicum)

**Tweet 1 (Hook)**
Can you beat Bitcoin DCA with a simple signal?

For my Georgia Tech practicum, I tried. Here's what I found — and what didn't work. 🧵

**Tweet 2 (The Failure)**
First attempt: train a CNN on candlestick chart images to predict next-day price direction.

Result? 41.43% on the tournament metric. A neutral baseline that buys the same amount every day scored 41.94%.

Pattern recognition ≠ trading edge.

**Tweet 3 (What Worked)**
What actually worked was embarrassingly simple:

• OLS trend slope of log price (60-day window)
• Z-score against 252-day rolling std
• Clip outliers at ±2σ
• Convert to daily purchase weight

Result: 44.95% — a +3.01 pp improvement over neutral DCA.

**Tweet 4 (The Insight)**
The key isn't prediction. It's regime awareness.

Bitcoin's volatility clusters. Bull and bear periods have statistically distinct return distributions (Mann-Whitney U, p < 0.001).

Normalizing the trend against its own recent volatility makes the signal automatically adaptive.

**Tweet 5 (Null Results)**
What didn't work:

• Volatility-amplitude scaling — redundant (z-score already captures vol)
• Weekly resampling — underperformed by -0.90 pp
• Proportional fees — mathematically percentile-invariant in this framework

Sometimes the null result IS the result.

**Tweet 6 (Takeaway)**
If you're DCA-ing into Bitcoin, you're already doing something sensible.

This research suggests a modest improvement: when the trend is strong relative to recent volatility, buy a little less. When it's weak, buy a little more. Not dramatically — just a tilt.

**Tweet 7 (Links)**
Full analysis + reproducibility package: https://github.com/Katzup/bitcoin-analytics-capstone-template
7-minute video walkthrough: https://youtu.be/dPQTMGjO_cM

Completed as part of Georgia Tech's OMSA Trilemma Practicum (Spring 2026).

#Bitcoin #QuantFinance #DataScience #GeorgiaTech #DCA
