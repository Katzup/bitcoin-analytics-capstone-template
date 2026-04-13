# VTS Final Report — Outline
# GT OMSA Practicum | Bob Katz | Spring 2026
# Status: SKELETON — fill in once sponsor requirements confirmed

---

## Section Map

| # | Section | Status | Notes |
|---|---------|--------|-------|
| 1 | Executive Summary | 🟡 Draft | ~300 words, key numbers |
| 2 | Problem Statement & Motivation | 🟡 Draft | DCA baseline, tournament framing |
| 3 | Methods — Midterm (CNN/GAF) | ⬜ TODO | Brief: what we built, why it failed |
| 4 | Methods — Post-Midterm Pivot | 🟡 Draft | OLS signal, z-score, governor |
| 5 | Results | 🟡 Draft | Ladder table, fee robustness, traditional metrics (5.5) |
| 6 | Ablation Studies | 🟡 Draft | Vol-amplitude, weekly granularity |
| 7 | Governance & Reproducibility | 🟡 Draft | EXECUTION_ASSUMPTIONS, smoke tests |
| 8 | Conclusions & Limitations | ⬜ TODO | Honest assessment of ceiling |
| Appendix | — | ⬜ TODO | Code listings, extended tables |

---

## Key Numbers (locked — do not change in report)

- **Neutral baseline**: 41.94% RW, 70.42% win rate
- **CNN (GAF)**: 41.43% RW — below neutral
- **OLS raw contrarian**: 43.25% RW (+1.31 pp vs neutral)
- **OLS z-score winsorized ±2σ**: 44.95% RW, 66.94% win rate **(+3.01 pp)**
- Evaluation: step=1, 3,076 rolling 365-day windows, 2016–2025
- **Fast-scan approximation** (step=7, 440 windows): 45.84% RW — NOT the headline

## Key Null Results (to include — strengthens methodology narrative)

- Vol-amplitude scaling: +0.14 pp RW, win rate ↓, SPD advantage ↓ → REJECTED
- Weekly granularity (12/26/52w): −0.90 pp → REJECTED
- Turnover governor: 0.00 pp RW impact, ~12% turnover reduction → ADOPTED (operational)
- Fee sensitivity: percentile-invariant in this harness (mathematical proof in report)

---

## Open TODOs (post sponsor confirmation)

- [ ] Confirm page / word limit
- [ ] Confirm section heading requirements (match rubric exactly)
- [ ] Confirm whether appendix counts toward limit
- [ ] Add citation for tournament metric (Trilemma repo / sponsor docs)
- [ ] Final figure: performance ladder bar chart (generate from PROJECT_STATE data)
