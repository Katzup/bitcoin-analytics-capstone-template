# Section 3: Methods — Midterm CNN/GAF Approach
# STATUS: DRAFT

---

## 3.1 Design Rationale

The midterm approach was motivated by the hypothesis that candlestick chart patterns encode actionable price momentum — a hypothesis supported by a substantial literature on deep learning for financial chart recognition. If a convolutional neural network could learn to classify recurring chart patterns by their historical forward-return distributions, the resulting probability estimates could directly parameterize the tournament allocator.

The core design decision was to represent 30-day BTC price windows as images rather than time series, enabling transfer of standard computer vision architectures without modification. The Gramian Angular Field (GAF) transform was selected because it preserves temporal ordering in a 2D image structure, avoiding the information loss of permutation-invariant transforms.

## 3.2 Gramian Angular Field Encoding

Given a 30-day close-price series rescaled to [−1, +1], the Gramian Angular Summation Field (GASF) is defined as:

$$G_{ij} = \cos(\phi_i + \phi_j), \quad \phi_i = \arccos(\tilde{x}_i)$$

where $\tilde{x}_i$ is the *i*-th rescaled price observation. The resulting 30×30 symmetric matrix encodes pairwise angular relationships between all time points. This encoding has the property that the main diagonal (i = j) recovers the original series, while off-diagonal terms capture interactions between non-adjacent time periods.

Each input window was preprocessed as:
1. Extract 30 consecutive daily closes from BTC-USD (Coinbase, 2014–2025)
2. Min-max scale to [−1, +1] within the window
3. Apply GASF transform → 30×30 float matrix
4. Stack GASF and optionally GADF (difference field) as 2-channel input

## 3.3 CNN Architecture and Training

The model used a lightweight 4-layer convolutional architecture:

| Layer | Config |
|-------|--------|
| Conv2D × 2 | 32 filters, 3×3, ReLU, BatchNorm |
| MaxPool | 2×2 |
| Conv2D × 2 | 64 filters, 3×3, ReLU, BatchNorm |
| MaxPool | 2×2 |
| Flatten + Dense | 128 units, ReLU, Dropout(0.5) |
| Output | 2-class softmax (up / down) |

**Label construction**: Each 30-day window was labeled by whether the closing price 30 days after the window end was higher (`up`) or lower/flat (`down`) than the final window close. This formulation targets 30-day directional classification, which aligns with a monthly accumulation horizon.

**Training protocol**:
- Train set: 2014–2015 (pre-institutional BTC era)
- Test set: 2016–2025 (tournament evaluation span)
- Optimizer: Adam (lr=1e-3), 50 epochs, early stopping on validation accuracy
- Class balancing: stratified sampling to handle the mild `up`-skew in BTC history

The model was implemented in PyTorch. A `features_simplified.py` fallback module (OLS-based, no torch dependency) was maintained in parallel throughout development to ensure tournament endpoint compatibility in grader environments without CUDA.

## 3.4 Signal-to-Allocation Mapping

CNN output `p_up` (predicted probability of 30-day up move) was mapped to daily allocation weights via:

```python
prob_up = cnn_output[:, 1]          # softmax probability of "up" class
# Passed to smoothed-EMA allocator in tournament_mode/weights.py
```

The allocator converts `prob_up` into daily weights using an exponential moving average smoother (α = 0.30), normalized to sum to 1. Days with higher predicted `p_up` received moderately higher weight; the smoothing prevented extreme concentration.

## 3.5 Midterm Results and Failure Analysis

**Midterm result**: **41.43% RW percentile** — 0.51 percentage points below the neutral DCA baseline (41.94%).

The CNN failed to outperform uniform DCA in the tournament evaluation. Post-hoc analysis identified three contributing factors:

**1. Dataset shift**: Training data (2014–2015) predates the institutional adoption era. The 2016–2025 evaluation period exhibits materially different volatility regimes, liquidity depth, and momentum structure. Patterns that were predictive in the pre-institutional regime may have reversed or disappeared.

**2. Classification accuracy ≠ trading edge**: Even if the CNN correctly predicted directional 30-day returns above chance, this accuracy would not necessarily translate to sats-per-dollar improvement. The tournament metric rewards *consistent within-window* relative cheapness, not raw directional prediction. A high-accuracy classifier that buys confidently on days that turn out to be local peaks — even if the 30-day return is positive — still loses on SPD.

**3. Signal granularity mismatch**: The CNN produces a single probability estimate per 30-day window, implying no intra-window weight variation. All days within a "predicted up" window receive the same upward tilt, regardless of whether that window's early days are priced at a discount relative to its late days. The tournament metric rewards exploiting intra-window price variation, which the CNN architecture structurally cannot capture.

**What the midterm result established**: The failure was informative rather than merely disappointing. It demonstrated that image-based pattern recognition on candlestick charts does not imply exploitable timing advantage under the tournament metric, and motivated a fundamental pivot to signal engineering that directly targets intra-window price dynamics — the OLS z-score approach described in Section 4.

**Alternative encodings considered but not tested**: Three alternative time-series image encodings — Markov Transition Field (MTF), Recurrence Plot (RP), and Wavelet scalogram — were identified as post-midterm candidates. MTF (available in the `pyts` library) captures state-transition probabilities between quantile bins and was estimated to offer +2–5 pp RW improvement over GAF; RP captures phase-space recurrence structure and was estimated at +1–4 pp; Wavelet scalograms provide multi-resolution time-frequency decomposition via `pywt` and were estimated at +3–6 pp. None were tested. The root cause analysis concluded that the failure modes were architectural rather than encoding-specific: (1) any binary directional classifier produces a single probability per window and cannot exploit *intra*-window price variation, which is precisely what the tournament metric rewards; (2) dataset shift from the 2014–2015 training regime to the 2016–2025 evaluation span would persist regardless of encoding choice; and (3) classification accuracy on 30-day direction does not map to sats-per-dollar improvement even if accuracy is high. Substituting MTF or RP for GASF would not have addressed these structural incompatibilities. The decision to redirect effort toward OLS-based signal engineering — which targets daily allocation weights directly — was therefore principled rather than a concession of scope.

---

*The torch model weights and training code are preserved in `tournament_mode/features.py`. The `features_simplified.py` fallback (OLS signal, no torch dependency) is the active implementation used in all post-midterm evaluation.*
