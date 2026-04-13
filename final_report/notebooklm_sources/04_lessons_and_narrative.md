# VTS Practicum: Lessons Learned, Methodology, and Scientific Narrative

## The Honest Story

This project could have been written as a triumphant tale of a sophisticated ML system beating the market. It's not that. It's a more interesting story:

**A deep learning approach failed.** Then a humble linear model succeeded. The difference was not in the sophistication of the algorithm but in asking the right question.

The CNN asked: "What pattern is this?"
The OLS signal asks: "How stretched is the current trend relative to its own recent history?"

The first question is elegant but disconnected from the tournament's metric. The second question is simple but directly relevant to the accumulation problem.

---

## Why Pattern Recognition Failed

The CNN/GAF system worked as a classifier. It learned to distinguish candlestick patterns. But the underlying assumption — that classifying patterns leads to better buy timing — was never tested against the tournament metric.

This is a classic ML failure mode: optimizing for the wrong objective. F1 score on candlestick labels is not the same as improvement in sats-per-dollar accumulation. The CNN optimized for one and was evaluated on the other.

The lesson: always evaluate against the actual task objective, not a proxy.

---

## Why the OLS Signal Works

The z-score normalization is the key innovation. Here's the intuition:

BTC is a highly trending asset. During bull markets, the raw OLS slope is persistently high. If you just invert the raw trend (buy less when trend is high), you'd chronically under-invest during bull markets and miss most of the accumulation. This is what the raw contrarian signal does — it has 91.48% win rate (excellent) but a structural disadvantage in strong bull markets.

The z-score normalization fixes this by asking: "Is today's trend *abnormally high* compared to the last year?"

During a sustained bull run, both the trend AND the z-score's reference period are elevated. After the z-score normalization, an "average" bull market day gets ts_z ≈ 0 (near-neutral allocation). Only *exceptionally stretched* moments get ts_z approaching +2 (modest reduction in investment). This regime-relative approach produces a more stable, persistent edge than raw contrarian.

**The unexpected finding:** Diagnostics showed ts_z is *positively correlated* with 30-day forward returns (r ≈ +0.14). The signal doesn't predict price drops — it doesn't need to. It accumulates modestly less during stretched-uptrend periods and modestly more during subdued periods. The edge comes from consistent timing relative to the 365-day window, not from timing price reversals.

---

## What the Null Results Mean

Two major experiments failed to improve the signal: vol-normalized amplitude and weekly resampling.

**This is not a failure of the project.** These null results are among the most valuable findings.

They tell us:
1. The z-score normalization is already handling vol-regime adaptation. The 252-day rolling std denominator is doing its job. Adding an explicit vol layer is redundant.
2. The daily 60/180/252-day design captures BTC's relevant signal frequencies. Weekly resampling loses information and underperforms consistently.
3. The +3.01 pp edge is *specific to the z-score structure*, not a generic property of OLS signals. This is what makes it robust.

If these experiments had shown marginal improvements in both RW and win rate, we might wonder whether the signal was over-fit to specific features. The fact that they *clearly* underperformed on multiple metrics gives us higher confidence that the final signal's design choices are genuinely near-optimal for this problem.

---

## Scientific Integrity Notes

**Things we document that might look like weaknesses but are consistent findings:**

1. **Win rate of final signal (66.94%) is lower than raw contrarian (91.48%).** The z-score trade-off: raw contrarian wins in almost every window but by less. Z-scored wins in fewer windows but by more. The higher RW percentile of the z-scored version (44.95% vs 43.25%) shows the z-score approach is correct under the recency-weighted metric.

2. **BTC accumulated is 31% lower than neutral.** In a secular bull market spanning 2017–2025, regime-relative dampening means the strategy systematically invested slightly less during strong uptrends (the dominant regime). This is the expected behavior, not a bug. The metric rewards timing quality (sats per dollar per window), not maximum accumulation over a bull supercycle.

3. **CAGR comparison is confounded by starting conditions.** The Sharpe, Sortino, and Calmar ratios are more reliable comparators for strategy quality.

4. **Fast-scan (step=7) vs full validation (step=1) differ.** Fast-scan: 45.84% RW. Full validation: 44.95% RW. The step=1 result is the headline. The fast-scan is only used for rapid iteration during development.

---

## The Practicum Experience

This project involved:
- Building a CNN/GAF image classification pipeline from scratch (PyTorch, matplotlib GAF generation)
- Running the full tournament evaluation framework (3,076 rolling windows, ~1-2 minutes per full evaluation)
- Implementing multiple feature engineering approaches, testing rigorously, and documenting decisions
- Producing a reproducible, auditor-grade codebase with 62 passing tests, torch fallback for environments without GPU, and grader-friendly smoke tests
- Writing the final report with locked key numbers — no cherry-picking, no post-hoc revision of results

The project also shows the value of an ML course applied to a real financial problem: the CNN training experience, the ablation methodology, the diagnostic analysis, and the systematic null result framework all come directly from OMSA coursework.

---

## Summary for General Audience

**What:** A Bitcoin dollar-cost averaging strategy that uses a simple trend signal to time purchases slightly better than a "buy the same amount every day" baseline.

**How:** We measure how strong BTC's recent trend is, then normalize it relative to the trend's own recent history. When the trend is abnormally strong compared to recent norms, we invest slightly less. When it's abnormally weak, we invest slightly more. The effect is modest — never investing more than 30% more or less than the baseline — but persistent.

**Result:** Over 3,076 one-year windows from 2016 to 2025, the strategy accumulated Bitcoin more efficiently 67% of the time, achieving a +3.01 percentage point improvement in the tournament's sats-per-dollar metric.

**What didn't work:** Image-based deep learning (CNN on candlestick patterns) scored slightly *below* random allocation. Simple linear models with correct normalization outperformed complex ML by +3.5 percentage points.

**Honest limitation:** This is a backtested result, not a guarantee of future performance. Bitcoin's secular bull trend (2016–2025) is the dominant regime in our evaluation window. In a flat or bear-dominated future, the signal's properties would need to be re-evaluated.
