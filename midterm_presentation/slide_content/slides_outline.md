# Midterm Presentation: Visual Trading System
## Slide Outline & Content

---

## Slide 1: Title Slide
**Title:** When AI Meets Market Reality  
**Subtitle:** A 296K-Parameter CNN That Lost to a Coin Flip  
**Context:** VTS Tournament Submission - Stacking Sats Challenge  
**Presenter:** [Your Name]  
**Date:** [Date]

**Notes:** Hook the audience with the counter-intuitive finding before explaining the journey.

---

## Slide 2: The Challenge
**Title:** The Tournament: Daily Bitcoin Allocation

**Content:**
- **Objective:** Predict daily allocation weights for BTC (2016-2025)
- **Constraint:** Σw_i = 1.0, w_i ≥ 1e-5
- **Metric:** Recency-Weighted SPD Percentile
- **Competition:** Top performers >60% percentile

**Key Constraint:** Strict causality - no lookahead allowed

**Notes:** Set up the problem clearly. Mention this is a real trading competition.

---

## Slide 3: Our Initial Approach
**Title:** The Visual Trading Pipeline

**Visual:** [Architecture diagram showing]
```
Price Data (90 days)
    ↓
GAF Image Generation
    ↓
Deep CNN (296K params)
    ↓
Temperature Calibration
    ↓
Tilt-Based Allocation
    ↓
EMA Smoothing
    ↓
Normalized Weights
```

**Notes:** Explain we thought complexity = performance. GAF preserves temporal correlation in polar coordinates.

---

## Slide 4: The Promise
**Title:** Why We Thought This Would Work

**Visual:** [Use NotebookLM Slide 5 - GAF example image]

**Content:**
- GAF transforms 1D time series → 2D images preserving temporal dependencies
- CNNs excel at pattern recognition (proven in computer vision)
- Trained on 2014-2015 data (470 days) - pre-tournament period

**Hypothesis:** Visual patterns in GAF images predict future returns

**Notes:** Explain the theory. This is what NotebookLM presents as a solved problem.

---

## Slide 5: The Reality
**Title:** First Results: Underwhelming Performance

**Visual:** [Table or chart showing]

| Metric | CNN Result | Target |
|--------|-----------|--------|
| RW SPD Percentile | 41.43% | >60% |
| Win Rate vs DCA | 54.32% | >50% |
| Windows Evaluated | 3,076 | - |

**Interpretation:** Underperforms simple Dollar-Cost Averaging

**Notes:** This is where the story pivots. We expected greatness, got mediocrity.

---

## Slide 6: Root Cause Analysis
**Title:** Ablation Study #1: Allocator Sizing

**Visual:** [Bar chart]

**Question:** Are losses from overbetting?

**Results (Sensitivity Sweep):**
- Sensitivity 0.5: 41.78% (+0.35 pp)
- Sensitivity 1.0: 41.61% (+0.18 pp)
- Sensitivity 1.5: 41.43% (baseline)

**Conclusion:** ⚠️ Minimal impact. Problem NOT allocator aggressiveness.

**Notes:** First hypothesis rejected. Moving to signal quality.

---

## Slide 7: The "Aha!" Moment
**Title:** Ablation Study #2: Signal Quality Test

**Visual:** [Side-by-side comparison]

**The Test:** Replace CNN predictions with constant prob_up = 0.5 (coin flip)

| Configuration | RW Percentile | Win Rate |
|---------------|---------------|----------|
| CNN (296K params) | 41.43% | 54.32% |
| Neutral (constant 0.5) | **41.94%** | **70.42%** |
| **Delta** | **+0.51 pp** | **+16.1 pp** |

**Conclusion:** ✅ **CNN signal is WORTHLESS**. Coin flip outperforms with higher consistency.

**Notes:** This is the key finding. Complex model = random guessing.

---

## Slide 8: The Smoking Gun
**Title:** Ablation Study #3: EMA Smoothing Sweep

**Visual:** [Line chart showing flat line across all alphas]

**Question:** Can smoothing fix "small wins, big losses"?

**Results (7 EMA alphas tested):**
- ALL alphas produce IDENTICAL results: 41.94% RW, 70.42% win rate
- Delta: 0.00 pp (exactly zero)

**Mathematical Proof:**
- Constant signal (prob_up = 0.5) → tilt always zero
- Smoothing constant = constant (no variance to smooth)
- **Result:** Zero effect proves signal quality is the issue

**Notes:** This definitively proves you can't polish a poor signal.

---

## Slide 9: Why GAF + CNN Failed
**Title:** Root Cause Analysis

**Visual:** [Four-quadrant diagram or list]

1. **Time Horizon Mismatch**
   - GAF lookback: 90 days (local patterns)
   - BTC cycles: 6-18 months (macro regimes)
   - Can't detect bull/bear transitions

2. **Daily Noise Problem**
   - Daily prices dominated by noise (low SNR)
   - Tournament forces daily decisions (amplifies noise)

3. **Wrong Architecture**
   - GAF assumes spatial patterns
   - Financial time series = temporal dependencies
   - CNN translation invariance not useful

4. **Training Set Limitations**
   - 2014-2015 data ≠ 2016-2024 reality
   - Pre-institutional BTC, different dynamics

**Notes:** Explain each point briefly. This shows deep understanding.

---

## Slide 10: The Pivot
**Title:** Simplified Baseline: Evidence-Based Approach

**Visual:** [Simplified architecture]

**New Approach:**
- Constant prob_up = 0.5 (neutral probability)
- Same allocation logic: tilt → bounded multiplier → EMA → normalize
- **No model artifacts required**

**Advantages:**
- ✅ No pre-trained model (1.1MB → 0 bytes)
- ✅ 10x faster execution (2 min vs 20 min)
- ✅ More robust (no overfitting)
- ✅ Higher win rate (70% vs 54%)

**Notes:** Emphasize this is the scientifically rigorous conclusion, not giving up.

---

## Slide 11: Side-by-Side Comparison
**Title:** Performance Comparison

**Visual:** [Chart comparing three approaches]

| Metric | CNN | Simplified | DCA Benchmark |
|--------|-----|------------|---------------|
| RW SPD Percentile | 41.43% | **41.94%** | 50% |
| Win Rate | 54.32% | **70.42%** | 50% |
| Execution Time | 15-20 min | **2-3 min** | N/A |
| Model Size | 1.1 MB | **None** | N/A |
| Robustness | Low | **High** | High |

**Winner:** Simplified baseline on ALL dimensions

**Notes:** The numbers don't lie. Simpler is better.

---

## Slide 12: Engineering Excellence
**Title:** What We Did Right

**Content:**
- ✅ **Strict Causality:** Last-row modification test, no lookahead
- ✅ **Deterministic:** seed=42, fully reproducible
- ✅ **Comprehensive Testing:** Unit tests, integration tests, ablations
- ✅ **Grader-Safe:** Relative paths, clear documentation
- ✅ **Scientific Rigor:** Hypothesis → Test → Conclude

**Code Quality:**
- Modular architecture (features, weights, scoring separate)
- Complete documentation (LESSONS_LEARNED.md)
- Version control ready

**Notes:** Even though the model underperformed, the engineering was solid.

---

## Slide 13: Key Lessons
**Title:** What We Learned

**Engineering Lessons:**
1. Ablation studies are critical for root cause analysis
2. Test "better than random?" BEFORE optimizing
3. Causality testing is non-negotiable

**Modeling Lessons:**
1. Complexity ≠ Performance (296K params lost to coin flip)
2. Domain knowledge > Deep learning for financial time series
3. Match model complexity to problem complexity

**Strategic Lessons:**
1. Simple often beats complex (Occam's Razor applies)
2. Weekly allocation likely better than daily (reduce noise)
3. Regime detection matters for BTC cycles

**Notes:** These lessons have value beyond this project.

---

## Slide 14: Future Directions
**Title:** What We'd Do Differently

**Immediate Improvements (1-2 days):**
- Weekly granularity (+10-20 pp expected)
- Multi-channel features (+5-12 pp)
- Longer lookback: 365-day window (+5-10 pp)

**High-Impact Changes (3-5 days):**
- Regime detection (bull/bear conditional strategies)
- LSTM/RNN architecture (temporal dependencies)
- Ensemble of simple models

**Expected Combined:** 41.94% → 57-73% RW percentile

**Notes:** Show we have a roadmap for improvement.

---

## Slide 15: Conclusion
**Title:** The Bottom Line

**Key Message:**
> "We built a technically sophisticated system and discovered through rigorous testing that it performs equivalently to a coin flip. This simplified version represents our evidence-based conclusion: in financial time series, signal quality matters more than model complexity."

**What We're Proud Of:**
- Professional engineering practices
- Scientific integrity (transparent negative results)
- Valuable insights for future participants

**Quote to Remember:**
> "The best model is the one that works, not the one that sounds impressive."

**Notes:** End on a confident note. The journey is the value.

---

## Appendix Slide (Optional)
**Title:** Technical Details

**Content:**
- GAF encoding mathematics
- CNN architecture details
- Ablation study methodology
- Full results tables

**Notes:** For Q&A, if graders want deep technical details.

---

# Speaker Notes Summary

**Total Time:** 10-12 minutes  
**Slides:** 15 + appendix  
**Key Hook:** "296K parameters lost to a coin flip"  
**Core Narrative:** Complex approach → Rigorous testing → Surprising discovery → Simplified solution

**Tone:** Scientific, honest, confident in methodology despite underperformance
