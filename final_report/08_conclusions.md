# Section 8: Conclusions & Limitations
# STATUS: DRAFT

---

## 8.1 Summary of Findings

This project demonstrates that a modest but statistically persistent timing advantage over uniform dollar-cost averaging is achievable in the Bitcoin accumulation tournament through regime-relative signal engineering. The final OLS z-score winsorized signal achieves **44.95% RW percentile** — a **+3.01 percentage point improvement over neutral DCA** — evaluated across 3,076 rolling 365-day windows spanning nine years of BTC price history (2016–2025).

The mechanism is regime-relative dampening rather than classical contrarian timing. The signal does not attempt to predict Bitcoin's direction or identify local price bottoms. Instead, it reduces allocation slightly when the market's trend is historically stretched relative to a 252-day rolling baseline — a posture that allows the strategy to avoid the most expensive days within each window at a rate sufficient to produce persistent SPD improvement. Diagnostic analysis confirmed the signal is positively correlated with forward returns (r ≈ +0.14 at 30d), which is consistent with momentum-dampening behavior rather than contrarian "buy the dip" logic.

The project's trajectory — CNN failure at midterm, transparent OLS pivot, systematic null result documentation — constitutes its secondary contribution. Two high-potential variants were explicitly tested against stated acceptance criteria:

- **Volatility-normalized amplitude scaling**: +0.14 pp RW, win rate −1.59 pp, sats/$ advantage −32 → **REJECTED**. ts_z's implicit vol-normalization (252d rolling std) is sufficient; explicit short-horizon amplitude scaling is redundant.
- **Weekly granularity resampling**: −0.90 pp RW for the direct weekly equivalent (12/26/52w) → **REJECTED**. The +0.26 pp observed at 12/26/26w reflects shorter z-score window size, not weekly resampling benefit.

These null results narrow the explanation of the +3.01 pp edge to the z-score normalization structure specifically and increase confidence that the result is not an artifact of design flexibility or post-hoc parameter tuning.

## 8.2 Performance Ceiling Assessment

The +3.01 pp improvement over neutral DCA represents a real but modest advantage. To contextualize: at 44.95% RW, the strategy ranks in the 45th percentile of the strategy distribution — meaningfully above the neutral baseline but well below competitive strategies that exploit more aggressive regime detection, multi-asset allocation, or machine learning features with richer information sets.

Several lines of evidence suggest this is close to the performance ceiling for a daily OLS signal class on this asset:

1. **Mechanism saturation**: ts_z already provides rolling volatility normalization. Explicit amplitude scaling and temporal resampling both failed to add incremental value beyond this, suggesting the signal's adaptive capacity is near exhaustion.

2. **Consistent win rate**: 66.94% of windows outperform neutral DCA. This reflects a real edge but also implies a 33% failure rate — concentrated in high-volatility windows where DCA's steady accumulation is genuinely hard to beat.

3. **Harness constraints**: The evaluation harness's proportional fee invariance (proven mathematically) means real-world transaction costs would not change the percentile ranking — but would shrink the absolute SPD advantage. At 40–45 bps round-trip, the absolute sats/$ advantage approaches zero.

Higher performance would likely require qualitatively different signal sources: on-chain metrics (exchange inflows, long/short ratio), macro regime detection (Fed policy cycles, risk-on/off indicators), or reinforcement learning approaches that optimize directly against the tournament objective. These remain future directions.

## 8.3 Limitations

**Single asset, single metric**: All results are specific to Bitcoin and to the sats-per-dollar accumulation metric. The signal construction — particularly the regime-relative z-score — may not generalize to other assets or optimization objectives (e.g., Sharpe ratio maximization, drawdown minimization).

**Secular bull market context**: The 2016–2025 evaluation period is predominantly a bull market with two major cycles (2017–2018 and 2020–2021). A strategy that reduces purchases during historically stretched uptrends benefits structurally from this context. Performance in a sustained bear or sideways market — where the signal would presumably increase purchases during protracted downtrends — is untested.

**Harness-specific fee invariance**: The mathematical proof that proportional fees are percentile-invariant applies specifically to the tournament's SPD-based evaluation harness. In a different evaluation framework (e.g., absolute return optimization), transaction costs would directly affect rankings. The "fee-invariant" characterization should not be generalized beyond this evaluation context.

**Single-strategy evaluation**: The reported RW percentile is computed against a theoretical strategy distribution (the rolling window's SPD distribution), not against a real population of competing submissions. The neutral baseline's 41.94% RW reflects this single-strategy evaluation, not median performance relative to an actual tournament population.

**Look-ahead-free signal construction**: The signal uses only information available at the time of allocation (rolling OLS on observed log-prices, rolling statistics computed over past windows). No future information is used. However, the evaluation framework itself is retrospective — the 3,076-window sweep examines historical performance. Out-of-sample performance in a prospective deployment is unknown.

## 8.4 Future Directions

The most promising extensions, in decreasing order of estimated impact:

1. **On-chain signal integration**: Exchange net position change, realized profit/loss ratio, and MVRV Z-score provide signals orthogonal to price momentum. These metrics have demonstrated leading-indicator properties at cycle turning points and would address the signal's primary weakness: inability to distinguish between structurally expensive markets (cycle top) and temporarily stretched markets (mid-cycle consolidation).

2. **Regime-conditional strategy switching**: A two-regime model (trend / mean-reversion) that switches between momentum-following and contrarian tilts based on on-chain or macro regime indicators could improve win rate in the approximately 33% of windows where the current signal underperforms.

3. **Direct objective optimization**: The tournament metric's specific functional form (recency-weighted percentile of SPD) could be used as the optimization target in a policy gradient or evolutionary strategy search, potentially discovering non-parametric allocation strategies that outperform OLS-based signal engineering.

4. **Multi-asset generalization**: Applying the regime-relative z-score framework to other volatile, cyclical assets (e.g., ETH, gold) would test whether the signal structure generalizes or is specific to Bitcoin's volatility regime characteristics.

---

*All code, results, and reproducibility instructions are available in the project repository. See Section 7 for governance and reproducibility details.*
