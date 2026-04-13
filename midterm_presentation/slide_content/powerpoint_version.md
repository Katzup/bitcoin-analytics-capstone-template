# PowerPoint Version: Slide-by-Slide Content

## Format Guidelines
- **Font:** Calibri or Arial
- **Title Size:** 36-44pt
- **Body Size:** 24-28pt
- **Color Scheme:** Dark blue background (#1a1a2e) with white text, green accents (#4ecca3)

---

## Slide 1: Title

**Title:** When AI Meets Market Reality

**Subtitle:** A 296K-Parameter CNN That Lost to a Coin Flip

**Footer:**
- VTS Tournament Submission - Stacking Sats Challenge
- Practicum Midterm Presentation
- [Your Name]

**Visual:** Dark gradient background, clean typography

---

## Slide 2: Why Visual Trading Systems? ⭐ NEW

**Title:** Why Visual Trading Systems?

**Subtitle:** From Classic Technical Analysis to Cutting-Edge AI — A Personal Journey

**Content (3 boxes with colored borders):**

### Box 1 (Green border): The Foundation (MBA Era)
- Reading *Magee & Edwards' Technical Analysis of Stock Trends*
- Learning how human traders "see" patterns quants miss
- Head-and-shoulders, trendlines, support levels

### Box 2 (Red border): The Discovery (OMSA Program)
- Research from JP Morgan's AI group + Professor Baich
- Former GT ML4T professor
- CNNs achieving 0.90+ F1 scores on pattern classification

### Box 3 (Yellow border): The Gap & The Test
- Pattern recognition ≠ Profitable trading
- Does it work under real causality constraints?
- Stacking Sats tournament as the testbed

**Visual:** Three colored boxes arranged horizontally or vertically

---

## Slide 3: The Tournament Challenge

**Title:** The Tournament Challenge

**Content (bullets with emojis):**

🎯 **Objective:** Predict daily BTC allocation weights (2016-2025)

📊 **Metric:** Recency-Weighted SPD Percentile

⚖️ **Constraints:** 
- Σwᵢ = 1.0
- wᵢ ≥ 1e-5

🔒 **Critical:** Strict causality — no lookahead allowed

🏆 **Target:** Top performers >60% percentile

**Visual:** Simple icons next to each bullet

---

## Slide 4: The Promise (Hypothesis)

**Title:** The Promise: Why We Expected This to Work

**Content:**

**Hypothesis:** Visual patterns in GAF images predict future returns

**The Pipeline:**
- GAF transforms 1D time series → 2D images
- Preserves temporal correlation in polar coordinates
- CNN excels at pattern recognition (proven in computer vision)
- Trained on 2014-2015 data (470 days, pre-tournament)

**Expected Win:** Complex deep learning > Simple baselines

**Visual:** Simple flowchart or diagram

---

## Slide 5: GAF Encoding ⭐ INSERT NOTEBOOKLM SLIDE 5

**Title:** Encoding: GAF / MTF Example

**Content:**
- Show NotebookLM Slide 5 (GAF/MTF comparison)
- One-liner: "We transformed daily BTC windows into GAF images"

**Visual:** INSERT IMAGE: notebooklm_gaf.png (screenshot from NotebookLM PDF)

---

## Slide 6: Our Pipeline

**Title:** Our Visual Trading Pipeline

**Content (monospace/code style):**

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

**Visual:** Code-style box with arrows

---

## Slide 7: First Results

**Title:** First Results: Reality Check

**Table:**

| Metric | CNN Result | Target |
|--------|-----------|--------|
| RW SPD Percentile | **41.43%** | >60% |
| Win Rate vs DCA | 54.32% | >50% |
| Windows Evaluated | 3,076 | — |

**Bottom text (large, red):**
> Underperforms simple Dollar-Cost Averaging

**Visual:** Table with red highlighting on the gap

---

## Slide 8: Ablation Study #1

**Title:** Ablation Study #1: Allocator Sizing

**Content:**

**Question:** Are losses from overbetting?

**INSERT IMAGE:** `images/01_ablation_sensitivity.png`

**Results (Sensitivity Sweep):**
- Sensitivity 0.5: 41.78% (+0.35 pp)
- Sensitivity 1.5: 41.43% (baseline)

**Conclusion (red text):**
⚠️ Minimal impact. Problem NOT allocator aggressiveness.

---

## Slide 9: Ablation Study #2 (THE KEY SLIDE) ⭐

**Title:** Ablation Study #2: The "Aha!" Moment

**Content:**

**Test:** Replace CNN with constant prob_up = 0.5 (coin flip)

**INSERT IMAGE:** `images/02_ablation_signal_quality.png`

**Results:**

| Approach | RW Percentile | Win Rate |
|----------|---------------|----------|
| CNN (296K params) | 41.43% | 54.32% |
| Neutral (constant 0.5) | **41.94%** | **70.42%** |

**Conclusion (green text):**
✅ CNN signal is WORTHLESS. Coin flip outperforms with higher consistency.

---

## Slide 10: Ablation Study #3

**Title:** Ablation Study #3: EMA Smoothing

**Content:**

**Question:** Can smoothing fix the performance?

**INSERT IMAGE:** `images/03_ablation_ema_flat.png`

**Results (7 alphas tested):**
- ALL alphas produce IDENTICAL results
- 41.94% RW, 70.42% win rate
- Delta: 0.00 pp

**Conclusion (red text):**
📉 Zero effect. Mathematical proof: signal quality is the issue.

---

## Slide 11: Why It Failed

**Title:** Why GAF + CNN Failed: Root Cause Analysis

**INSERT IMAGE:** `images/06_root_causes.png`

**Key Points:**
1. Time Horizon Mismatch (90-day vs 6-18 month cycles)
2. Daily Noise Problem (low SNR)
3. Wrong Architecture (spatial vs temporal)
4. Training Set Issues (2014-2015 ≠ 2016-2024)
5. Overengineering (296K params for binary outcome)

---

## Slide 12: The Pivot

**Title:** The Evidence-Based Pivot

**Content:**

**Simplified Baseline:**
- Constant prob_up = 0.5 (neutral probability)
- Same allocation logic
- **NO model artifacts required**

**Comparison:**

| Metric | CNN | Simplified |
|--------|-----|------------|
| RW Percentile | 41.43% | **41.94%** ⭐ |
| Win Rate | 54.32% | **70.42%** ⭐ |
| Execution Time | ~17 min | **~2.5 min** ⭐ |
| Model Size | 1.1 MB | **None** ⭐ |

**INSERT IMAGE:** `images/04_performance_comparison.png`

---

## Slide 13: Engineering Excellence

**Title:** What We Did Right

**Two columns:**

### Technical Excellence
✓ Strict causality enforcement  
✓ Deterministic (seed=42)  
✓ Comprehensive testing  
✓ Grader-safe implementation  

### Scientific Rigor
✓ Hypothesis-driven development  
✓ Systematic ablation studies  
✓ Transparent negative results  
✓ Learning orientation  

**Visual:** Two-column layout with checkmarks

---

## Slide 14: Key Lessons

**Title:** Key Lessons Learned

**Content (numbered):**

1. **Ablation studies are critical** — Without them, we'd never know CNN was worthless

2. **Test "better than random?" early** — Should be Ablation #0, not #2

3. **Complexity ≠ Performance** — 296K parameters lost to coin flip

4. **Simple often beats complex** — Occam's Razor applies to trading

**Bottom quote:**
> "The best model is the one that works, not the one that sounds impressive."

---

## Slide 15: Future Work

**Title:** Future Improvements

**INSERT IMAGE:** `images/07_future_improvements.png`

**Highest ROI:**
- Weekly granularity (+10-20 pp expected)
- Regime detection (+15-25 pp)
- Multi-channel features (+5-12 pp)

**Expected Combined:** 41.94% → 57-73% RW percentile

---

## Slide 16: Conclusion

**Title:** The Bottom Line

**Quote (large, italic):**
> "We built a technically sophisticated system and discovered through rigorous testing that it performs equivalently to a coin flip. This simplified version represents our evidence-based conclusion: in financial time series, signal quality matters more than model complexity."

**Bottom text:**
🎓 Scientific Integrity > Raw Performance

---

## Slide 17: Thank You / Q&A

**Title:** Thank You

**Subtitle:** Questions?

**Resources:**
- Documentation: LESSONS_LEARNED.md
- Code: Visual_Trading_System repository
- Notebook: btc_accumulation_model_simplified.ipynb

**Final Quote:**
> "The best model is the one that works, not the one that sounds impressive."

---

## Images Checklist

| Slide | Image File | Source |
|-------|-----------|--------|
| 5 | notebooklm_gaf.png | Screenshot from NotebookLM PDF Slide 5 |
| 8 | 01_ablation_sensitivity.png | Provided |
| 9 | 02_ablation_signal_quality.png | Provided ⭐ |
| 10 | 03_ablation_ema_flat.png | Provided |
| 11 | 06_root_causes.png | Provided |
| 12 | 04_performance_comparison.png | Provided |
| 15 | 07_future_improvements.png | Provided |

---

## Timing Guide

| Slide Range | Content | Time |
|-------------|---------|------|
| 1-2 | Title + Why This Project | 2 min |
| 3-6 | Tournament + Approach | 2 min |
| 7 | Results | 1 min |
| 8-10 | Ablation Studies (CORE) | 3 min |
| 11-12 | Analysis + Pivot | 2 min |
| 13-15 | Engineering + Lessons + Future | 2 min |
| 16-17 | Conclusion + Q&A | 1 min |
| **Total** | | **13 min** |

---

## Notes for PowerPoint Assembly

1. **Slide 5:** Take screenshot of NotebookLM PDF Slide 5 (GAF/MTF comparison)
2. **Color consistency:** Use green (#4ecca3) for positive/wins, red (#ff6b6b) for warnings/failures
3. **Font consistency:** Use same font throughout (Calibri recommended)
4. **Image sizing:** Charts should fill 60-70% of slide width
5. **Animation:** Consider fade-in for ablation results to build suspense
