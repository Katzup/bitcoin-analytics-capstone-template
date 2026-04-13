# Tournament Scoring Plan (FINAL - Score-Identical)

**Source**: Official tournament template verification + user clarifications
**Date**: 2026-01-21
**Status**: AUTHORITATIVE - Use this for all implementations

---

## Scoring Per 12-Month Window

Given a 12-month window of daily prices `p_t` and daily weights `w_t`:

### 1. Compute "inv_price" (satoshis per dollar)
```python
inv_price_t = (1.0 / p_t) * 1e8
```

### 2. Uniform DCA SPD (baseline)
```python
uniform_spd = mean(inv_price_t)
```
Arithmetic mean because uniform weights imply `w_t = 1/N`.

### 3. Strategy SPD
```python
dynamic_spd = sum(w_t * inv_price_t)
```
Equivalent to: `1e8 * sum(w_t / p_t)`.

### 4. Percentile within window (both baseline and strategy)
```python
min_spd = min(inv_price_t)
max_spd = max(inv_price_t)
span = max_spd - min_spd
uniform_pct = (uniform_spd - min_spd) / span * 100
dynamic_pct = (dynamic_spd - min_spd) / span * 100
```

**Interpretation**: Percentile is a 0–100 normalization between "all buys on worst day" (0%) and "all buys on best day" (100%) for that window.

---

## Aggregation Across Rolling Windows

**Date Range**: 2016-01-01 to 2025-06-01 (3,075 rolling 12-month windows)

Compute `dynamic_pct[i]` for every rolling window.

### Primary Tournament Metric: RW_SPD_PCT

**Recency-weighted average percentile** using exponential decay:

```python
decay = 0.9
N = len(dynamic_pct)
raw_w = np.array([decay ** (N - 1 - i) for i in range(N)])
exp_w = raw_w / raw_w.sum()
RW_SPD_PCT = sum(dynamic_pct[i] * exp_w[i])
```

This is the official template's `exp_avg_pct` variable.

### Secondary Metric: Win Rate

**Pass ratio** (fraction of windows beating DCA):

```python
wins = sum(1 for i in range(N) if dynamic_pct[i] > uniform_pct[i])
pass_ratio = wins / N
```

---

## Final "Score" for Ranking

**Revised Stance**: Don't hardcode a composite until confirmed in official leaderboard.

### Compute and Report BOTH:
1. `RW_SPD_PCT` (recency-weighted percentile) - **PRIMARY**
2. `pass_ratio` (win rate) - **SECONDARY**

### For Internal Tracking
```python
primary_score = RW_SPD_PCT  # Main ranking statistic per official template
```

**Note**: If tournament leaderboard later reveals a composite formula (e.g., 50% RW + 50% WR), we can add it without breaking compatibility.

---

## Weight Constraints (Tournament Rules)

```python
MIN_WEIGHT = 1e-5
```

### Requirements
- `w_t >= MIN_WEIGHT` for all days
- `sum(w_t) == 1.0` within tolerance: `np.isclose(total, 1.0, rtol=1e-5, atol=1e-8)`

### Enforcement Algorithm (Deterministic)
```python
def enforce_constraints(raw_weights: np.ndarray) -> np.ndarray:
    """
    Three-step process to enforce constraints.

    Step 1: Normalize raw weights
    Step 2: Clamp to minimum
    Step 3: Re-normalize to sum=1
    """
    # Step 1: Normalize
    w = raw_weights / raw_weights.sum()

    # Step 2: Clamp to minimum (may cause sum > 1)
    w = np.maximum(w, MIN_WEIGHT)

    # Step 3: Re-normalize to exactly 1.0
    w = w / w.sum()

    return w
```

---

## Validation Checklist (Score-Critical)

### 1. Reference Equivalence Tests

**Test 1: Uniform weights should match DCA exactly**
```python
w_uniform = np.ones(N) / N
assert dynamic_spd(w_uniform) == uniform_spd
assert dynamic_pct(w_uniform) == uniform_pct
```

**Test 2: Constant price should produce identical percentiles**
```python
# If all prices are same, any weight vector should give 50% percentile
prices_constant = np.ones(N) * 30000
assert abs(dynamic_pct(any_weights) - 50.0) < 1e-6
```

### 2. Causality / No-Lookahead Test

**Last-row modification test**:
```python
def test_causality(compute_weights_fn):
    df_original = create_test_window(365)
    weights_original = compute_weights_fn(df_original)

    # Modify last row drastically
    df_modified = df_original.copy()
    df_modified.iloc[-1, df_modified.columns.get_loc('prob_up')] = 0.99
    weights_modified = compute_weights_fn(df_modified)

    # First N-1 weights must be unchanged
    assert np.allclose(weights_original.iloc[:-1], weights_modified.iloc[:-1]), \
        "CAUSALITY VIOLATION: Earlier weights changed"
```

### 3. Official Scorer Parity

**For a small sample window, verify**:
```python
assert abs(our_spd - official_spd) < 1e-6
assert abs(our_pct - official_pct) < 1e-6
assert abs(our_RW - official_exp_avg_pct) < 1e-6
```

---

## Reference Implementation

### Complete SPD Calculation
```python
def calculate_spd_for_window(
    weights: pd.Series,
    prices: pd.Series
) -> dict:
    """
    Calculate SPD metrics for a single 12-month window.

    Returns:
        dict with keys:
            - dynamic_spd: Strategy SPD
            - uniform_spd: DCA baseline SPD
            - dynamic_pct: Strategy percentile (0-100)
            - uniform_pct: DCA percentile (0-100)
            - min_spd: Worst-case SPD (all buys on peak)
            - max_spd: Best-case SPD (all buys on bottom)
    """
    # Validate inputs
    assert abs(weights.sum() - 1.0) < 1e-5, f"Weights sum: {weights.sum()}"
    assert len(weights) == len(prices), "Length mismatch"

    # Step 1: Compute inv_price (sats per dollar)
    inv_price = (1.0 / prices) * 1e8

    # Step 2: Uniform DCA SPD (baseline)
    uniform_spd = inv_price.mean()

    # Step 3: Strategy SPD
    dynamic_spd = (weights * inv_price).sum()

    # Step 4: Percentile calculations
    min_spd = inv_price.min()
    max_spd = inv_price.max()
    span = max_spd - min_spd

    uniform_pct = (uniform_spd - min_spd) / span * 100 if span > 0 else 50.0
    dynamic_pct = (dynamic_spd - min_spd) / span * 100 if span > 0 else 50.0

    return {
        'dynamic_spd': dynamic_spd,
        'uniform_spd': uniform_spd,
        'dynamic_pct': dynamic_pct,
        'uniform_pct': uniform_pct,
        'min_spd': min_spd,
        'max_spd': max_spd
    }
```

### Recency-Weighted Percentile
```python
def calculate_recency_weighted_percentile(
    percentiles: np.ndarray,
    decay: float = 0.9
) -> float:
    """
    Calculate RW_SPD_PCT (primary tournament metric).

    Args:
        percentiles: Array of dynamic_pct values (oldest first, newest last)
        decay: Exponential decay rate (default 0.9 from official template)

    Returns:
        Recency-weighted average percentile
    """
    N = len(percentiles)

    # Oldest (i=0) gets smallest weight, newest (i=N-1) gets weight=1.0
    raw_w = np.array([decay ** (N - 1 - i) for i in range(N)])

    # Normalize to sum=1
    exp_w = raw_w / raw_w.sum()

    # Weighted average
    RW_SPD_PCT = (percentiles * exp_w).sum()

    return RW_SPD_PCT
```

### Win Rate
```python
def calculate_win_rate(
    dynamic_percentiles: np.ndarray,
    uniform_percentiles: np.ndarray
) -> float:
    """
    Calculate pass_ratio (secondary tournament metric).

    Args:
        dynamic_percentiles: Strategy percentiles
        uniform_percentiles: DCA baseline percentiles

    Returns:
        Win rate (fraction of windows beating DCA)
    """
    wins = (dynamic_percentiles > uniform_percentiles).sum()
    pass_ratio = wins / len(dynamic_percentiles)

    return pass_ratio
```

---

## Implementation Priority

1. ✅ **Specification complete** (this document)
2. → **Implement scoring.py** with exact formulas above
3. → **Write unit tests** for all three validation categories
4. → **Implement compute_weights()** with causality enforcement
5. → **Implement rolling window evaluator** (3,075 windows)
6. → **Verify against official scorer** on sample data

---

## Leaderboard Clarification (Action Item)

**TODO**: Check tournament page/repo for exact leaderboard ranking formula.

**Possible outcomes**:
- **A**: Leaderboard ranks by RW_SPD_PCT only → we're done
- **B**: Leaderboard uses composite (e.g., 0.5*RW + 0.5*WR) → add composite function
- **C**: Leaderboard uses multiple categories → report both metrics separately

**Current stance**: Implement both RW_SPD_PCT and pass_ratio, use RW_SPD_PCT as primary until confirmed otherwise.
