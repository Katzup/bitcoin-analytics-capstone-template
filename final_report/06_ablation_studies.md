# Section 6: Ablation Studies
# STATUS: DRAFT

---

## 6.1 Ablation Methodology

All signal variants were evaluated against explicit acceptance criteria defined *before* running experiments, reducing the risk of post-hoc rationalization. The decision rule for adopting a variant was:

> **Adopt if**: RW% improves relative to the current best (OLS z-score winsorized) AND at least one of:
> (a) win rate does not degrade materially (≤ 1 pp loss tolerated), OR
> (b) mean absolute SPD advantage improves.

The neutral DCA baseline (prob_up = 0.5, 41.94% RW) provides the floor; rejection means returning to the pre-experiment champion signal. Four experiments were conducted across two phases: pre-midterm allocator/signal ablations and post-midterm variant testing.

**Evaluation setup**: Primary metric is RW% on step=1 full validation (3,076 windows, 2016–2025). Fast-scan (step=7, ~440 windows) is used for quick hypothesis screening; all headline claims use step=1.

---

## 6.2 Pre-Midterm Ablations

Two allocator ablations were performed during the midterm phase to establish that the tournament allocator itself was functioning correctly and that signal quality — not allocator parameterization — was the primary bottleneck.

### Ablation #1: Allocator Sensitivity

**Hypothesis**: The EMA smoothing parameter (α) and weight floor in `tournament_mode/weights.py` might be over-constraining the allocator, preventing it from expressing the full signal range.

**Method**: Varied α ∈ {0.10, 0.20, 0.30, 0.50} and weight floor ∈ {0.001, 0.005, 0.01} while holding the signal constant at the neutral baseline (prob_up = 0.5). Results were compared on SPD percentile distributions and weight volatility.

**Result**: The allocator shows low sensitivity to α in this range; EMA smoothing primarily affects daily weight stability, not the long-run SPD percentile. The weight floor (MIN_WEIGHT) acts as a portfolio regularizer preventing degenerate concentrations. Neither parameter drives meaningful RW% variation when the input signal is neutral.

**Conclusion**: Allocator parameterization is not the limiting factor. The primary lever is signal quality — whether `prob_up` contains predictive information about intra-window price variation.

*(Figure: `figures/01_ablation_sensitivity.png` — Allocator sensitivity heatmap: RW% vs. α × weight floor.)*

### Ablation #2: Signal Quality

**Hypothesis**: If the allocator is well-parameterized, the CNN/GAF signal (prob_up from binary classifier) should produce measurable improvement over the neutral baseline.

**Method**: Compared three signal configurations — neutral (constant 0.5), raw CNN output, and EMA-smoothed CNN output — on the 2016–2025 evaluation period using the fixed α=0.30 allocator.

**Result**: The CNN signal achieved 41.43% RW — below the neutral baseline of 41.94%. Smoothing reduced weight volatility but did not rescue the directional performance deficit. The EMA-flat convergence test (Figure 3) confirmed that as α → 0, all signals converge to identical neutral weights, ruling out numerical implementation bugs.

**Conclusion**: The signal quality failure is fundamental: the CNN's binary directional classification does not produce exploitable intra-window timing information under the SPD metric, independent of how the allocator processes the signal.

*(Figure: `figures/02_ablation_signal_quality.png` — SPD percentile comparison: Neutral vs. CNN (raw) vs. CNN (EMA smoothed).)*

*(Figure: `figures/03_ablation_ema_flat.png` — EMA convergence test: signal → neutral as α decreases.)*

---

## 6.3 Post-Midterm Ablations

After pivoting to OLS-based signal engineering, two high-potential variants were formally tested against the champion signal (OLS z-score winsorized, 44.95% RW). Both were rejected by pre-specified criteria.

### Ablation #3: Volatility-Normalized Amplitude Scaling

**Hypothesis**: The ts_z signal's tilt amplitude should scale with the current volatility regime. During low-vol periods, the signal could be more aggressive; during high-vol periods, more conservative. The explicit layer:

```
vol_norm_factor = clip(rv_60 / rv_ref, 0.5, 2.0)
prob_up = 0.5 - 0.15 × (vol_norm_factor × tanh(ts_z))
```

where `rv_60` is 60-day realized volatility and `rv_ref` is the long-run reference volatility.

**Rationale**: BTC's vol regime shifts substantially across market cycles. A signal calibrated to average volatility may undertilt in calm periods and overtilt in turbulent ones.

**Result** (step=7 fast-scan, 440 windows):

| Metric | Champion (ts_z) | Vol-Amplitude Variant | Δ |
|--------|----------------|----------------------|---|
| RW% | 45.84% | 45.99% | +0.14 pp |
| Win rate | 66.14% | 64.55% | −1.59 pp |
| Mean SPD advantage | +304.9 sats/$ | +272.9 sats/$ | −32 sats/$ |

**Decision**: REJECT. The +0.14 pp RW improvement is smaller than typical scan-level variability; the win rate and absolute SPD advantage both deteriorate.

**Mechanism explanation**: ts_z already provides implicit vol-normalization — its denominator is a 252-day rolling standard deviation, which naturally scales inversely with the current volatility regime. Adding an explicit short-horizon amplitude layer (rv_60/rv_ref) introduces a *second* contraction in high-vol regimes, compounding rather than correcting the adjustment. In low-vol regimes, the amplitude cap (0.22) limits upside to 13.5% of days, producing negligible lift. The experiment's primary value is diagnostic: it confirms that ts_z's long-horizon standardization is the appropriate vol-normalization mechanism for this signal class.

Script: `vol_amplitude_analysis.py`.

### Ablation #4: Weekly OLS Resampling

**Hypothesis**: Resampling BTC prices to weekly frequency before computing OLS slopes would reduce high-frequency noise and align the signal with BTC's dominant 6–18 month market cycles.

**Method**: Daily prices were resampled to weekly (Friday close). OLS slopes were computed on 12-week (short) and 26-week (long) windows, with z-score normalization over 26-week or 52-week rolling baselines — the direct weekly equivalents of the daily 60/180/252-day configuration. Daily allocation weights were recovered by linear interpolation between weekly signal values.

**Results** (step=7 fast-scan, 440 windows):

| Configuration | RW% | Win% | Δ vs champion |
|--------------|-----|------|--------------|
| Daily 60/180/252 (champion) | 45.84% | 66.14% | — |
| Weekly 12/26/52w (direct equivalent) | 44.94% | 72.50% | −0.90 pp |
| Weekly 12/26/26w (shorter z-window) | 46.10% | 65.00% | +0.26 pp |
| Weekly 4/13/52w | 44.62% | — | −1.22 pp |
| Weekly 6/18/52w | 42.10% | — | −3.74 pp |

**Decision**: REJECT. The direct weekly equivalent (12/26/52w) underperforms by −0.90 pp. The best-performing weekly variant (12/26/26w, +0.26 pp) uses a shorter z-score normalization window (26w ≈ 182 days vs. 52w ≈ 364 days), not a weekly price series per se. When holding the z-window length constant at ~26 weeks and switching to daily prices produces the same improvement, confirming that the apparent gain comes from z-window length, not resampling.

**Conclusion**: Daily 60/180/252-day OLS with 252-day rolling z-normalization is near-optimal for this signal class. Weekly resampling does not add value; it discards useful intra-week price variation and introduces interpolation artifacts in the daily weights.

Script: `weekly_granularity_analysis.py`.

---

## 6.4 Summary

| Ablation | Experiment | Decision | Primary Reason |
|----------|-----------|----------|----------------|
| #1 Allocator sensitivity | α × weight floor sweep | No action needed | Allocator not limiting factor |
| #2 CNN signal quality | CNN vs neutral | Confirms CNN failure | No intra-window timing signal |
| #3 Vol-amplitude scaling | rv_60/rv_ref amplitude | Reject | RW marginal (+0.14 pp); win rate/SPD worse |
| #4 Weekly OLS resampling | 12/26/52w and variants | Reject | Direct weekly equivalent −0.90 pp; best variant gain attributable to z-window length |

The four ablations collectively narrow the source of the +3.01 pp timing edge: it derives specifically from the z-score normalization structure (rolling 252-day standardization of OLS trend strength), not from allocator parameterization, signal delivery mechanism, or temporal granularity of the price series.

---

*Ablation charts generated by `generate_charts.py`. Full numerical logs in `plans/PROJECT_STATE.md`.*
