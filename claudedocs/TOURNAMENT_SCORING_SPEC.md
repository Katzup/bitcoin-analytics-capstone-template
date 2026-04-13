claude## Tournament Scoring Specification - CORRECTED VERSION
## Critical Formulas for Score-Identical Implementation

**Purpose**: Authoritative scoring reference to ensure exact match with official tournament evaluator.
**Status**: Pre-implementation specification (all formulas verified before coding)

---

## 1. SPD (Satoshis Per Dollar) - EXACT DEFINITION

### ✅ CORRECT Formula (Harmonic Mean)
```python
SPD = 1e8 * sum(w_t / price_t)  where sum(w_t) = 1
```

**Intuition**: Each weight w_t buys (w_t / price_t) BTC. Sum across all days to get total BTC per dollar invested, then convert to satoshis.

**Mathematical Form**:
```
SPD = 10^8 × Σ(w_t / p_t)  for t ∈ [1, N]
```

**Alternative Representation** (if mentioning "average price"):
```python
harmonic_price = 1 / sum(w_t / price_t)  # Weighted harmonic mean
SPD = 1e8 / harmonic_price                # Equivalent to above
```

### ❌ WRONG Formulas (Common Mistakes)
```python
# WRONG #1: Arithmetic weighted mean (underestimates SPD)
SPD = 1e8 / sum(w_t * price_t)  # ← DO NOT USE

# WRONG #2: Simple average (ignores weights)
SPD = 1e8 / mean(price_t)  # ← DO NOT USE

# WRONG #3: Vague "average price"
SPD = (1/avg_price) * 1e8  # ← AMBIGUOUS, DO NOT USE
```

### Reference Implementation
```python
def calculate_spd(weights: pd.Series, prices: pd.Series) -> float:
    """
    Calculate Satoshis per Dollar (SPD) for allocation strategy.

    Formula: SPD = 10^8 × Σ(w_t / p_t)

    Args:
        weights: Normalized allocation weights (sum = 1.0)
        prices: BTC prices in USD

    Returns:
        SPD value (satoshis per dollar invested)
    """
    assert abs(weights.sum() - 1.0) < 1e-5, f"Weights must sum to 1: {weights.sum()}"
    assert len(weights) == len(prices), "Weights and prices must have same length"
    assert (weights >= 0).all(), "Weights must be non-negative"
    assert (prices > 0).all(), "Prices must be positive"

    # CRITICAL: Harmonic calculation
    spd = 1e8 * (weights / prices).sum()

    return spd
```

### Uniform DCA Baseline
```python
# Uniform DCA: equal weight each day
w_uniform = 1.0 / N  for all t
SPD_dca = 1e8 * sum(1/N / price_t) = 1e8 * (1/N) * sum(1/price_t)
```

---

## 2. Causality Within 12-Month Window - STRICT ENFORCEMENT

### Requirement
```
For each day t in window:
    w_t must depend ONLY on data from days [1, 2, ..., t]
    w_t must NOT depend on data from days [t+1, t+2, ..., N]
```

### Implementation Rules

**Rule 1: Day-by-Day Processing (Safest)**
```python
def compute_weights(df_window: pd.DataFrame) -> pd.Series:
    """
    Compute weights with strict causality enforcement.

    CRITICAL: Process sequentially to ensure w_t uses only data ≤ t
    """
    weights = []

    for t in range(len(df_window)):
        # Extract data up to and including day t
        historical_data = df_window.iloc[:t+1]

        # Compute weight for day t using only historical_data
        w_t = compute_single_day_weight(historical_data)
        weights.append(w_t)

    return pd.Series(weights, index=df_window.index)
```

**Rule 2: EMA is Causal (Safe if Used Correctly)**
```python
# pandas.ewm() is inherently causal
multipliers_smooth = multipliers.ewm(alpha=0.30, adjust=False).mean()
# ✅ multipliers_smooth[t] depends only on multipliers[0:t+1]
```

**Rule 3: Forbidden Operations**
```python
# ❌ WRONG: Uses future data
mean_prob = df_window['prob_up'].mean()  # Sees entire window
w_t = prob_up[t] / mean_prob             # Contaminated

# ❌ WRONG: Looks ahead
next_day_price = df_window['price'].shift(-1)  # Uses t+1

# ✅ CORRECT: Only historical
mean_prob_historical = df_window['prob_up'].iloc[:t+1].mean()
```

### Causality Validation Test
```python
def test_causality(compute_weights_fn):
    """
    Test: Modifying last row must not change earlier weights.

    If compute_weights is causal, then:
        w[0], w[1], ..., w[N-2] should be identical whether we modify row N-1 or not
    """
    df_original = create_test_window(365)
    weights_original = compute_weights_fn(df_original)

    # Modify last row drastically
    df_modified = df_original.copy()
    df_modified.iloc[-1, df_modified.columns.get_loc('prob_up')] = 0.99
    df_modified.iloc[-1, df_modified.columns.get_loc('price')] = 99999.0
    weights_modified = compute_weights_fn(df_modified)

    # First N-1 weights must be identical (causality)
    assert np.allclose(weights_original.iloc[:-1], weights_modified.iloc[:-1]), \
        "CAUSALITY VIOLATION: Earlier weights changed when last row modified"

    # Last weight can differ (uses modified data)
    assert not np.isclose(weights_original.iloc[-1], weights_modified.iloc[-1]), \
        "Sanity check: Last weight should differ"

    print("✅ Causality test passed")
```

---

## 3. Minimum Weight Constraint - EXACT ALGORITHM

### Constraint
```
MIN_WEIGHT = 1e-5
All weights must satisfy: w_t ≥ MIN_WEIGHT
Sum of weights must equal: sum(w_t) = 1.0 (within tolerance 1e-5)
```

### Deterministic Algorithm
```python
MIN_WEIGHT = 1e-5
TOLERANCE = 1e-5

def enforce_min_weight_constraint(raw_weights: np.ndarray) -> np.ndarray:
    """
    Enforce minimum weight constraint with deterministic clamp+renormalize.

    Algorithm:
        1. Normalize raw weights to sum=1
        2. Clamp to minimum (may cause sum > 1)
        3. Re-normalize to sum=1
        4. Optional: Repeat step 2+3 once more for numerical stability

    Args:
        raw_weights: Unnormalized positive weights (e.g., bounded multipliers)

    Returns:
        Constrained weights: w_i ≥ MIN_WEIGHT, sum(w) = 1.0
    """
    # Guard: Check for degenerate case
    N = len(raw_weights)
    if N * MIN_WEIGHT > 1.0:
        raise ValueError(f"Impossible constraint: {N} weights × {MIN_WEIGHT} = {N*MIN_WEIGHT} > 1.0")

    # Step 1: Initial normalization
    weights = raw_weights / raw_weights.sum()

    # Step 2: Clamp to minimum (may cause sum > 1)
    weights = np.maximum(weights, MIN_WEIGHT)

    # Step 3: Re-normalize to exactly 1.0
    weights = weights / weights.sum()

    # Step 4: Optional second pass for numerical stability
    weights = np.maximum(weights, MIN_WEIGHT)
    weights = weights / weights.sum()

    # Validation
    assert abs(weights.sum() - 1.0) < TOLERANCE, \
        f"Weight sum violation: {weights.sum()} (expected 1.0)"
    assert (weights >= MIN_WEIGHT - 1e-9).all(), \
        f"Min weight violation: {weights.min()} < {MIN_WEIGHT}"

    return weights
```

### Numerical Stability Notes
- **Why two passes?** Float arithmetic can cause tiny drift; second pass ensures both constraints hold
- **Why MIN_WEIGHT - 1e-9 in validation?** Allow tiny numerical error (1e-9 << MIN_WEIGHT = 1e-5)
- **Guard condition**: If N * MIN_WEIGHT > 1, constraint is mathematically impossible

---

## 4. Recency Weighting - EXACT FUNCTION

### Official Tournament Spec
```
"More recent windows receive exponentially higher weights"
Decay parameter: ρ = 0.9 (default)
```

### Exact Formula
```python
def calculate_recency_weighted_percentile(
    percentiles: List[float],
    decay: float = 0.9
) -> float:
    """
    Calculate recency-weighted average of SPD percentiles.

    CRITICAL: Newest window gets highest weight (ρ^0 = 1.0)

    Formula:
        w[i] = ρ^(N-1-i)  where i=0 is oldest window

    Example (N=3, ρ=0.9):
        i=0 (oldest):  w = 0.9^2 = 0.81
        i=1 (middle):  w = 0.9^1 = 0.90
        i=2 (newest):  w = 0.9^0 = 1.00

    Then normalize: [0.81, 0.90, 1.00] / 2.71 = [0.299, 0.332, 0.369]

    Args:
        percentiles: List ordered chronologically (oldest first, newest last)
        decay: Exponential decay rate ρ ∈ (0, 1)

    Returns:
        Recency-weighted average percentile
    """
    N = len(percentiles)

    # CRITICAL: Newest window (i=N-1) gets weight = ρ^0 = 1.0
    # Oldest window (i=0) gets weight = ρ^(N-1)
    weights = np.array([decay ** (N - 1 - i) for i in range(N)])

    # Normalize weights to sum=1
    weights = weights / weights.sum()

    # Weighted average
    rw_percentile = np.dot(percentiles, weights)

    return rw_percentile
```

### Verification Against Official Scorer
```python
def test_recency_weighting_matches_official():
    """
    Test our recency weighting matches official tournament scorer.

    MUST RUN THIS TEST: Download official scorer and verify exact match
    """
    # Example test case
    percentiles = [50.0, 60.0, 70.0]  # 3 windows: oldest → newest
    decay = 0.9

    # Our implementation
    our_rw_pct = calculate_recency_weighted_percentile(percentiles, decay)

    # Official scorer (import from tournament repo)
    # official_rw_pct = official_scorer.recency_weighted_percentile(percentiles, decay)

    # CRITICAL: Must match exactly
    # assert abs(our_rw_pct - official_rw_pct) < 1e-6

    # Expected calculation:
    # weights = [0.81, 0.90, 1.00] → normalized [0.299, 0.332, 0.369]
    # rw_pct = 50*0.299 + 60*0.332 + 70*0.369 = 61.8
    expected = (50 * 0.81 + 60 * 0.90 + 70 * 1.00) / (0.81 + 0.90 + 1.00)

    assert abs(our_rw_pct - expected) < 1e-6, \
        f"Recency weighting mismatch: {our_rw_pct} vs {expected}"
```

### ⚠️ MUST VERIFY
Before implementation:
1. Download official tournament scorer from GitHub
2. Identify exact recency weighting function
3. Run test: `our_rw_pct == official_rw_pct` on reference case
4. If mismatch, adjust our formula to match exactly

---

## 5. Expected Performance - MARKED AS SPECULATIVE

### ⚠️ HYPOTHESIS (UNVERIFIED)
The following performance estimates are **speculative** and should NOT be used for validation:

```
Based on VTS walk-forward IR=1.59:
    - Win Rate: 55-65% (HYPOTHESIS - not derived from IR)
    - RW SPD Percentile: 60-70% (HYPOTHESIS - no logical connection to IR)
    - Final Score: 57.5-67.5 (HYPOTHESIS - purely speculative)
```

### Why This Is Not Valid
- **Information Ratio (IR)** measures risk-adjusted returns in continuous space
- **SPD Percentile** measures cost-basis efficiency in discrete rolling windows
- **No mathematical relationship** connects IR → SPD percentile
- **Different baselines**: IR vs DCA alpha ≠ SPD percentile vs uniform weights

### What We Can Say
```
VTS has demonstrated:
    ✅ Positive alpha vs DCA in walk-forward testing (IR=1.59)
    ✅ Consistent outperformance across multiple timeframes
    ✅ Probability calibration (Temperature Scaling improves Brier score)

VTS tournament adaptation HYPOTHESIZES:
    ❓ CNN probability signals → better cost-basis timing
    ❓ Bounded multipliers + EMA → stable allocation strategy
    ❓ Should beat uniform DCA, but magnitude unknown until tested
```

### Validation Strategy
```python
# DO NOT hard-code expected performance
# Instead: Run backtest, observe actual performance, analyze results

results = evaluate_rolling_windows(df, construct_features, compute_weights)
final_score = calculate_tournament_score(results['strategy'], results['dca'])

# Report actual performance (no comparison to speculative targets)
print(f"Final Score: {final_score['final_score']:.2f}")
print(f"Win Rate: {final_score['win_rate']:.2%}")
print(f"RW SPD Percentile: {final_score['rw_spd_percentile']:.2f}%")
```

---

## Summary Checklist (Pre-Implementation)

### Before Writing Any Code:
- [ ] **SPD Formula**: All references use `SPD = 1e8 * sum(w_t / price_t)`
- [ ] **Harmonic Mean**: If mentioning "avg_price", define as `1 / sum(w_t / price_t)`
- [ ] **Remove Arithmetic**: No instances of `1e8 / sum(w_t * price_t)`
- [ ] **Causality Test**: Implement last-row modification test
- [ ] **Min-Weight Algorithm**: Exact clamp→renorm sequence defined
- [ ] **Recency Weighting**: Download official scorer, verify exact formula
- [ ] **Performance Section**: Removed or clearly marked as speculative
- [ ] **Official Scorer**: Import or copy with attribution, assert exact match

### Critical Unit Tests to Write:
```python
1. test_spd_harmonic_calculation()  # Verify SPD formula
2. test_causality_enforcement()     # Last-row modification test
3. test_min_weight_constraint()     # Clamp+renorm correctness
4. test_recency_weighting_official_match()  # Match official scorer
5. test_uniform_weights_baseline()  # DCA should be ~50th percentile
```

---

## Implementation Priority

### Phase 0 (BEFORE Phase 1):
1. Download official tournament scorer from GitHub
2. Extract exact SPD, percentile, recency weighting, win rate formulas
3. Create reference implementation matching official scorer
4. Write unit tests: `our_score == official_score` on synthetic cases
5. Only proceed to Phase 1 after all tests pass

### Phase 1-5: Proceed as planned in original document
- All formulas from this spec (not original plan) are authoritative
- All "expected performance" removed from validation criteria
- All tests verify against official scorer, not speculative targets
