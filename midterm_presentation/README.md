# VTS Midterm Presentation - Assembly Guide

## 📁 What Was Generated

This directory contains everything needed to assemble your midterm presentation:

```
midterm_presentation/
├── README.md                              # This file
├── slide_content/
│   └── slides_outline.md                  # Full slide-by-slide content
├── images/                                # Generated chart assets
│   ├── 01_ablation_sensitivity.png       # Ablation #1: Sensitivity sweep
│   ├── 02_ablation_signal_quality.png    # Ablation #2: CNN vs Neutral (KEY SLIDE)
│   ├── 03_ablation_ema_flat.png          # Ablation #3: EMA flat line
│   ├── 04_performance_comparison.png     # Side-by-side comparison
│   ├── 05_temporal_pattern.png           # U-shaped performance over time
│   ├── 06_root_causes.png                # Why GAF+CNN failed
│   └── 07_future_improvements.png        # ROI analysis for future work
└── generate_charts.py                     # Python script to regenerate charts
```

---

## 🎯 How to Assemble Your Presentation

### Option 1: PowerPoint / Google Slides (Recommended)

1. **Create a new presentation** (16:9 aspect ratio)

2. **Add slides using the outline** in `slide_content/slides_outline.md`
   - Copy-paste titles and bullet points
   - Insert corresponding images from `images/` folder

3. **Slide-by-slide mapping:**

| Slide # | Title | Content Source | Image Asset |
|---------|-------|----------------|-------------|
| 1 | Title Slide | Custom | None |
| 2 | The Challenge | slides_outline.md | None |
| 3 | Our Initial Approach | slides_outline.md | Create simple diagram |
| 4 | The Promise | slides_outline.md | Use NotebookLM Slide 5 |
| 5 | The Reality | slides_outline.md | Simple table |
| 6 | Ablation #1 | slides_outline.md | `01_ablation_sensitivity.png` |
| 7 | Ablation #2 (KEY) | slides_outline.md | `02_ablation_signal_quality.png` ⭐ |
| 8 | Ablation #3 | slides_outline.md | `03_ablation_ema_flat.png` |
| 9 | Why It Failed | slides_outline.md | `06_root_causes.png` |
| 10 | The Pivot | slides_outline.md | `04_performance_comparison.png` |
| 11 | Side-by-Side | slides_outline.md | Same as #10 or new layout |
| 12 | Engineering Wins | slides_outline.md | None (bullet points) |
| 13 | Key Lessons | slides_outline.md | None (bullet points) |
| 14 | Future Work | slides_outline.md | `07_future_improvements.png` |
| 15 | Conclusion | slides_outline.md | None (quote) |

### Option 2: Markdown-to-Slides (Marp/Reveal.js)

If you prefer code-based presentations:

1. **Install Marp** (VS Code extension) or use Reveal.js
2. **Convert** `slides_outline.md` to Marp format
3. **Insert images** using `![alt](images/xxxxx.png)`
4. **Export** to PDF or HTML

### Option 3: Jupyter Notebook as Presentation

1. **Open** `btc_accumulation_model_simplified.ipynb`
2. **Add markdown cells** between code cells using slide content
3. **Use RISE extension** for Jupyter to present as slides
4. **Export** to slides.html

---

## 🖼️ Image Assets Reference

### Key Visuals to Use

#### 1. `02_ablation_signal_quality.png` ⭐ MOST IMPORTANT
- **Use for:** Slide 7 (The "Aha!" Moment)
- **Shows:** CNN vs Neutral comparison
- **Message:** 296K parameters = random guessing

#### 2. `04_performance_comparison.png`
- **Use for:** Slide 10-11
- **Shows:** Four-dimension comparison
- **Message:** Simplified wins on ALL metrics

#### 3. `03_ablation_ema_flat.png`
- **Use for:** Slide 8
- **Shows:** Flat line across all EMA alphas
- **Message:** Mathematical proof of signal quality issue

#### 4. `06_root_causes.png`
- **Use for:** Slide 9
- **Shows:** Why GAF+CNN failed
- **Message:** Time horizon mismatch, noise, wrong architecture

---

## 📝 Speaker Notes Summary

### The Hook (Slides 1-2)
> "We built a 296K-parameter deep learning system... and discovered it performs equivalently to a coin flip."

### The Pivot (Slides 6-8)
> "Through systematic ablation testing, we proved the CNN signal was worthless. The neutral baseline achieved better performance with zero model complexity."

### The Conclusion (Slide 15)
> "Sometimes the most valuable outcome is discovering what doesn't work - and having the scientific integrity to document it."

---

## 🎨 Design Suggestions

### Color Scheme
- **Primary:** Dark blue (#2c3e50) - Professional, trustworthy
- **Accent:** Green (#2ecc71) - Success, simplified approach
- **Highlight:** Orange (#e67e22) - Key findings
- **Warning:** Red (#e74c3c) - Failures, negative results

### Fonts
- **Headings:** Sans-serif (Arial, Helvetica, or Calibri)
- **Body:** Same family, smaller size
- **Code/Numbers:** Monospace for metrics

### Layout Tips
1. **One idea per slide** - Don't crowd
2. **Large fonts** - Minimum 24pt for body, 36pt for titles
3. **High contrast** - Dark text on light background
4. **Consistent positioning** - Keep titles in same spot

---

## ⏱️ Timing Guide

**Total: 10-12 minutes**

| Slide Range | Content | Time |
|-------------|---------|------|
| 1-2 | Introduction | 1 min |
| 3-5 | Approach & Initial Results | 2 min |
| 6-8 | Ablation Studies (CORE) | 3 min |
| 9-11 | Analysis & Pivot | 2 min |
| 12-14 | Engineering & Lessons | 2 min |
| 15 | Conclusion | 1 min |
| - | Q&A Buffer | 1-2 min |

---

## 📊 What Makes This Presentation Strong

1. **Scientific Rigor** - Three systematic ablation studies
2. **Honest Reporting** - Transparent about negative results
3. **Clear Narrative** - Problem → Testing → Discovery → Solution
4. **Quantified Results** - Specific metrics, not vague claims
5. **Engineering Pride** - Good practices despite model underperformance

---

## 🔧 Regenerating Charts

If you need to modify charts:

```bash
cd /Users/bobkatz/Visual_Trading_System/midterm_presentation
source ../.pdf_env/bin/activate
python generate_charts.py
```

Edit `generate_charts.py` to change colors, data, or add new charts.

---

## ✅ Pre-Presentation Checklist

- [ ] All 15 slides created
- [ ] 7 chart images inserted in correct slides
- [ ] NotebookLM Slide 5 (GAF) included
- [ ] Speaker notes reviewed
- [ ] Timing practiced (10-12 min total)
- [ ] Backup PDF exported
- [ ] QR code for Audio Overview (optional)

---

## 📎 Additional Resources to Include

1. **LESSONS_LEARNED.md** - Link in appendix or handout
2. **EXECUTIVE_SUMMARY.md** - For graders who want details
3. **NotebookLM Audio Overview** - QR code for deep dive
4. **Git repository** - Show code and commit history

---

**Good luck with your midterm! The story of discovery is compelling - own it.**
