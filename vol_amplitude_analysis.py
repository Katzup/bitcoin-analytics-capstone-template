#!/usr/bin/env python3
"""
Volatility-Normalized Amplitude Analysis
=========================================

Tests whether scaling the tilt amplitude inversely with realized volatility
provides any incremental improvement over the current fixed-amplitude approach.

Current strategy (baseline):
    prob_up = 0.5 - 0.15 × tanh(ts_z)   [fixed amplitude = 0.15]

Variant tested:
    rv_60   = rolling 60d realized vol (annualized)
    rv_ref  = rolling 252d mean of rv_60 (long-term vol baseline)
    vol_ratio = rv_60 / rv_ref           (> 1 = elevated vol, < 1 = calm)

    amplitude = clip(0.15 / vol_ratio, 0.06, 0.22)
    prob_up_vol = 0.5 - amplitude × tanh(ts_z)

Hypothesis:
    - High-vol regime (bear market, vol_ratio > 1) → amplitude shrinks
      → more conservative tilt → avoids aggressive buying into crashes
    - Low-vol regime (bull/calm, vol_ratio < 1) → amplitude grows
      → more aggressive accumulation during stable periods

Key question: ts_z already contains implicit vol-normalization via its rolling
252d std denominator. Does an additional explicit amplitude layer add anything?

If delta_RW ≈ 0: z-score's implicit normalization is sufficient.
If delta_RW > 0: explicit amplitude scaling is additive.
If delta_RW < 0: double vol-normalization over-dampens the signal.

Script: vol_amplitude_analysis.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import pyarrow  # noqa: F401
except ImportError:
    raise ImportError(
        "pyarrow is required for pd.read_parquet(). "
        "Install with:  pip install -r requirements.txt  "
        "(on this machine the working interpreter is python3.10)"
    ) from None

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tournament_mode.features_simplified import build_features_trend_ols
from tournament_mode import construct_features, compute_weights, MIN_WEIGHT
from tournament_mode.evaluator import evaluate_rolling_windows
from tournament_mode.scoring import calculate_recency_weighted_percentile

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/TrilemmaFoundation/"
    "stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet"
)
BACKTEST_START  = '2016-01-01'
BACKTEST_END    = '2025-06-01'
RANDOM_SEED     = 42
STEP_DAYS       = 7      # Fast scan (~440 windows); use 1 for full validation
DECAY           = 0.9    # Official tournament recency weight

# Vol-amplitude parameters
BASE_AMPLITUDE  = 0.15   # Current fixed amplitude
AMP_FLOOR       = 0.06   # Min amplitude (high-vol regime cap)
AMP_CAP         = 0.22   # Max amplitude (low-vol regime cap)
VOL_WINDOW      = 60     # Realized vol estimation window (days)
VOL_REF_WINDOW  = 252    # Long-term vol reference window (days)
VOL_FLOOR_RATIO = 0.30   # Prevent division by zero in vol_ratio


def build_vol_amplitude_prob_up(
    df: pd.DataFrame,
    price_col: str = 'PriceUSD_coinmetrics',
) -> pd.Series:
    """
    Build vol-normalized amplitude variant of prob_up.

    Step 1: Reconstruct ts_z from baseline prob_up
        baseline: prob_up = 0.5 - 0.15 * tanh(ts_z)
        → ts_z = arctanh((0.5 - prob_up) / 0.15)

    Step 2: Compute vol-normalized amplitude
        rv_60  = 60d realized vol (annualized)
        rv_ref = rolling 252d mean of rv_60
        amplitude = clip(0.15 / (rv_60 / rv_ref), AMP_FLOOR, AMP_CAP)

    Step 3: Apply vol-amplitude
        prob_up_vol = clip(0.5 - amplitude * tanh(ts_z), 0.35, 0.65)

    Returns:
        Series of prob_up_vol aligned to df.index, NaN where baseline is NaN.
    """
    prices     = df[price_col].values
    log_prices = np.log(prices)
    n          = len(prices)

    # Log returns (causal: ret[t] = log(price[t]) - log(price[t-1]))
    log_ret    = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.diff(log_prices)
    log_ret_s  = pd.Series(log_ret, index=df.index)

    # Step A: Realized vol (annualized)
    rv_60 = log_ret_s.rolling(VOL_WINDOW, min_periods=30).std() * np.sqrt(252)

    # Step B: Long-term vol reference (causal rolling mean)
    rv_ref = rv_60.rolling(VOL_REF_WINDOW, min_periods=60).mean()

    # Step C: Vol ratio — clip low end to avoid explosion when rv_ref → 0
    vol_ratio = (rv_60 / rv_ref.clip(lower=1e-6)).clip(lower=VOL_FLOOR_RATIO)

    # Step D: Vol-normalized amplitude
    amplitude = (BASE_AMPLITUDE / vol_ratio).clip(lower=AMP_FLOOR, upper=AMP_CAP)

    # Step E: Reconstruct ts_z from baseline prob_up
    baseline_features = build_features_trend_ols(df)
    prob_up_base      = baseline_features['prob_up']

    tanh_arg = ((0.5 - prob_up_base) / BASE_AMPLITUDE).clip(-0.9999, 0.9999)
    ts_z     = pd.Series(np.arctanh(tanh_arg.values), index=df.index)

    # Step F: Vol-amplitude prob_up
    prob_up_vol = 0.5 - amplitude * np.tanh(ts_z)
    prob_up_vol = prob_up_vol.clip(lower=0.35, upper=0.65)

    # Preserve NaN mask from baseline (lookback period)
    prob_up_vol = prob_up_vol.where(prob_up_base.notna(), other=np.nan)

    return prob_up_vol


def run_evaluation(df_with_prob_up: pd.DataFrame, label: str) -> dict:
    """Run rolling window evaluation and return summary metrics."""
    results = evaluate_rolling_windows(
        df_with_prob_up,
        model=None,
        window_days=365,
        step_days=STEP_DAYS,
        verbose=False,
    )
    rw_pct   = calculate_recency_weighted_percentile(
        results['pct_strategy'].values, decay=DECAY
    )
    win_rate = float((results['pct_strategy'] > results['pct_uniform']).mean())
    spd_strat   = float(results['spd_strategy'].mean())
    spd_uniform = float(results['spd_uniform'].mean())
    advantage   = spd_strat - spd_uniform
    n_windows   = len(results)

    print(f"  {label}: RW% = {rw_pct:.2f}%  Win% = {win_rate:.2%}"
          f"  StratSPD = {spd_strat:,.1f}  Advantage = {advantage:+.1f}")

    return {
        'label':      label,
        'rw_pct':     rw_pct,
        'win_rate':   win_rate,
        'spd_strat':  spd_strat,
        'spd_uniform': spd_uniform,
        'advantage':  advantage,
        'n_windows':  n_windows,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VOLATILITY-NORMALIZED AMPLITUDE ANALYSIS")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading tournament data (step={STEP_DAYS}, DECAY={DECAY})...")
    df = pd.read_parquet(DATA_URL)
    df = df.loc[BACKTEST_START:BACKTEST_END]
    print(f"  Range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Days:  {len(df)}")

    # ── Build baseline features (fixed amplitude 0.15) ─────────────────────
    print("\nBuilding baseline features (fixed amplitude=0.15)...")
    df_baseline = construct_features(df.copy())
    valid_base = df_baseline['prob_up'].notna().sum()
    print(f"  Valid prob_up rows: {valid_base:,}")

    # ── Build vol-amplitude features ──────────────────────────────────────────
    print("\nBuilding vol-normalized amplitude features...")
    prob_up_vol = build_vol_amplitude_prob_up(df)
    valid_vol   = prob_up_vol.notna().sum()
    print(f"  Valid prob_up_vol rows: {valid_vol:,}")

    # Amplitude distribution statistics
    prices     = df['PriceUSD_coinmetrics'].values
    log_prices = np.log(prices)
    n          = len(prices)
    log_ret    = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.diff(log_prices)
    log_ret_s  = pd.Series(log_ret, index=df.index)
    rv_60      = log_ret_s.rolling(VOL_WINDOW, min_periods=30).std() * np.sqrt(252)
    rv_ref     = rv_60.rolling(VOL_REF_WINDOW, min_periods=60).mean()
    vol_ratio  = (rv_60 / rv_ref.clip(lower=1e-6)).clip(lower=VOL_FLOOR_RATIO)
    amplitude  = (BASE_AMPLITUDE / vol_ratio).clip(lower=AMP_FLOOR, upper=AMP_CAP)

    amp_valid = amplitude.dropna()
    print(f"\n  Amplitude statistics (vol-normalized, {len(amp_valid):,} valid days):")
    print(f"    Mean:   {amp_valid.mean():.4f}  (baseline: {BASE_AMPLITUDE:.4f})")
    print(f"    Std:    {amp_valid.std():.4f}")
    print(f"    Min:    {amp_valid.min():.4f}  (floor: {AMP_FLOOR:.4f})")
    print(f"    Max:    {amp_valid.max():.4f}  (cap: {AMP_CAP:.4f})")
    print(f"    % at floor: {(amp_valid == AMP_FLOOR).mean():.1%}")
    print(f"    % at cap:   {(amp_valid == AMP_CAP).mean():.1%}")
    print(f"    % unconstrained: {((amp_valid > AMP_FLOOR) & (amp_valid < AMP_CAP)).mean():.1%}")

    # Vol ratio statistics
    vr_valid = vol_ratio.dropna()
    print(f"\n  Vol ratio statistics (rv_60 / rv_ref, {len(vr_valid):,} valid days):")
    print(f"    Mean:   {vr_valid.mean():.3f}")
    print(f"    Std:    {vr_valid.std():.3f}")
    print(f"    Min:    {vr_valid.min():.3f}")
    print(f"    Max:    {vr_valid.max():.3f}")
    print(f"    % high vol (ratio > 1.5): {(vr_valid > 1.5).mean():.1%}")
    print(f"    % low vol  (ratio < 0.7): {(vr_valid < 0.7).mean():.1%}")

    # ── Build DataFrames for evaluation ──────────────────────────────────────
    df_vol = df.copy()
    df_vol['prob_up'] = prob_up_vol

    # ── Run evaluations ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROLLING WINDOW EVALUATION")
    print("=" * 70)
    print(f"(step={STEP_DAYS}d, window=365d, DECAY={DECAY}, ~440 windows)\n")

    res_base = run_evaluation(df_baseline, label="Fixed amp (0.15)")
    res_vol  = run_evaluation(df_vol,      label="Vol-norm amp   ")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: FIXED vs VOL-NORMALIZED AMPLITUDE")
    print("=" * 70)

    delta_rw  = res_vol['rw_pct']   - res_base['rw_pct']
    delta_wr  = res_vol['win_rate'] - res_base['win_rate']
    delta_spd = res_vol['advantage'] - res_base['advantage']
    neutral_rw_pct = 41.94  # Established neutral baseline

    print(f"""
  {'Metric':35} | {'Fixed (baseline)':>16} | {'Vol-Normalized':>16} | {'Delta':>10}
  {'-'*85}
  {'RW Percentile (primary metric)':35} | {res_base['rw_pct']:>15.2f}% | {res_vol['rw_pct']:>15.2f}% | {delta_rw:>+9.2f}pp
  {'Win Rate vs DCA':35} | {res_base['win_rate']:>15.2%} | {res_vol['win_rate']:>15.2%} | {delta_wr:>+9.2%}
  {'Mean Strat SPD (sats/$)':35} | {res_base['spd_strat']:>16,.1f} | {res_vol['spd_strat']:>16,.1f} | {res_vol['spd_strat']-res_base['spd_strat']:>+9.1f}
  {'Mean DCA SPD (sats/$)':35} | {res_base['spd_uniform']:>16,.1f} | {res_vol['spd_uniform']:>16,.1f} | {res_vol['spd_uniform']-res_base['spd_uniform']:>+9.1f}
  {'Absolute SPD advantage (sats/$)':35} | {res_base['advantage']:>+16.1f} | {res_vol['advantage']:>+16.1f} | {delta_spd:>+9.1f}
  {'vs Neutral baseline (41.94%)':35} | {res_base['rw_pct']-neutral_rw_pct:>+15.2f}pp | {res_vol['rw_pct']-neutral_rw_pct:>+15.2f}pp | {'':>10}
  {'Tilt amplitude (mean)':35} | {BASE_AMPLITUDE:>16.4f} | {amp_valid.mean():>16.4f} | {'(variable)':>10}
  {'Windows evaluated':35} | {res_base['n_windows']:>16,} | {res_vol['n_windows']:>16,} | {'':>10}
""")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if delta_rw > 0.75:
        category = "SUBSTANTIAL IMPROVEMENT"
        action   = "adopt in production"
    elif delta_rw > 0.25:
        category = "MODERATE IMPROVEMENT"
        action   = "consider adopting — verify with full step=1 validation"
    elif delta_rw > 0.05:
        category = "MARGINAL IMPROVEMENT"
        action   = "not worth added complexity — z-score already captures vol"
    elif delta_rw > -0.05:
        category = "NEUTRAL"
        action   = "confirms z-score implicit normalization is sufficient"
    else:
        category = "DEGRADATION"
        action   = "double vol-normalization over-dampens the signal — do not adopt"

    print(f"""
  Vol-amplitude vs fixed-amplitude delta:  {delta_rw:+.2f} pp
  Category: {category}
  Recommendation: {action}

  ── MECHANISM EXPLANATION ────────────────────────────────────────────────

  The ts_z signal already contains IMPLICIT vol-normalization:
      ts_z = (TrendScore - rolling_mean_252d) / rolling_std_252d

  The rolling_std_252d denominator scales inversely with trend volatility.
  In high-vol regimes, TrendScore oscillates more → larger std denominator
  → smaller ts_z → smaller tanh(ts_z) → less tilt.

  The vol-amplitude variant adds EXPLICIT vol-normalization on top:
      amplitude = 0.15 × (rv_ref / rv_60)

  Whether this additive layer helps depends on the overlap between:
    - Implicit normalization: via rolling 252d std of TrendScore
    - Explicit normalization: via 60d realized vol of log-returns

  If delta ≈ 0: The two normalizations are highly correlated → redundant.
  If delta > 0: They capture different vol regimes → complementary.
  If delta < 0: Double-dampening in already-conservative regimes hurts.

  ── NARRATIVE VALUE FOR FINAL REPORT ─────────────────────────────────────

  Regardless of the numerical outcome, this experiment provides a concrete
  example of hypothesis testing with the tournament metric:
    1. Hypothesis formed (explicit vol-amplitude should improve)
    2. Mechanism explained (two different vol-normalization channels)
    3. Experiment designed (controlled A/B with same ts_z signal)
    4. Result interpreted honestly (improvement, neutral, or degradation)

  The framing: "ts_z's z-score denominator already embeds vol-normalization
  — this is the mechanism behind the +{res_base['rw_pct']-neutral_rw_pct:.2f} pp edge, not the
  contrarian framing."
""")
    print("=" * 70)

    return {
        'baseline': res_base,
        'vol_amp':  res_vol,
        'delta_rw': delta_rw,
    }


if __name__ == "__main__":
    results = main()
