#!/usr/bin/env python3
"""
Weekly Granularity Analysis
============================

Hypothesis: Resampling daily BTC data to weekly reduces short-term noise
and better aligns the OLS trend signal with Bitcoin's characteristic
6-18 month market cycles.

Daily baseline (current best):
    short_window=60d, long_window=180d, z_window=252d
    prob_up = 0.5 - 0.15 × tanh(ts_z)   → 44.95% RW, 66.94% win rate

Weekly variant:
    Weekly prices: resample('W-MON').last() — one price per Mon-ending week
    Slopes annualized by ×52 (weeks per year, vs ×365 for daily)
    Same signal formula: prob_up = 0.5 - 0.15 × tanh(ts_z)
    Forward-fill back to daily index for evaluation

Parameter sweep (5 configurations):
    ( 8w / 26w / 52w)  — short lookback
    (12w / 26w / 52w)  — primary: maps to daily 84d / 182d / 364d
    (12w / 52w / 52w)  — longer long window (~1yr)
    (26w / 52w / 52w)  — both windows stretched
    (12w / 26w / 26w)  — shorter z-score window

Causality guarantee:
    resample('W-MON').last() → each week labeled on its closing Monday
    Signal at Monday T uses prices up to Monday T (inclusive)
    Forward-fill: daily D ∈ [Mon T, Mon T+1) uses signal from Mon T
    No look-ahead: Mon T signal is never applied before Mon T arrives

Decision rule (same as all prior experiments):
    Adopt if:  ΔRW > 0  AND  (win_rate stable OR SPD advantage improves)
    Reject if: ΔRW marginal (<0.25 pp) with degraded win_rate and SPD edge

Script: weekly_granularity_analysis.py
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

from tournament_mode.features_simplified import _fast_rolling_ols
from tournament_mode import construct_features
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
PRICE_COL       = 'PriceUSD_coinmetrics'
AMPLITUDE       = 0.15   # Same fixed amplitude as daily baseline
NEUTRAL_RW_PCT  = 41.94  # Established neutral baseline

# Weekly window parameter grid
# Each dict: short/long/zscore in weeks, plus a display label
WINDOW_GRID = [
    {'short':  8, 'long': 26, 'zscore': 52, 'label': " 8/26/52w"},
    {'short': 12, 'long': 26, 'zscore': 52, 'label': "12/26/52w"},   # primary
    {'short': 12, 'long': 52, 'zscore': 52, 'label': "12/52/52w"},
    {'short': 26, 'long': 52, 'zscore': 52, 'label': "26/52/52w"},
    {'short': 12, 'long': 26, 'zscore': 26, 'label': "12/26/26w"},
]


def build_weekly_ols_prob_up(
    df: pd.DataFrame,
    short_weeks: int = 12,
    long_weeks: int = 26,
    zscore_weeks: int = 52,
    amplitude: float = AMPLITUDE,
    price_col: str = PRICE_COL,
) -> pd.Series:
    """
    Build weekly-granularity OLS prob_up, forward-filled to daily.

    Steps:
        1. Resample daily prices → weekly (last price of Mon-ending week)
        2. Rolling OLS slopes on weekly log prices
        3. Annualize: slopes × 52 (weeks/year)
        4. TrendScore = 0.4*(short_slope × R²_short) + 0.6*(long_slope × R²_long)
        5. Z-score normalize over rolling zscore_weeks window (clip ±2σ)
        6. prob_up = 0.5 - amplitude × tanh(ts_z)
        7. Forward-fill weekly signal to daily index

    Causality:
        resample('W-MON').last() → signal at Monday T uses data up to Monday T.
        Forward-fill: every day D ∈ [Mon T, Mon T+1) inherits Monday T signal.
        No look-ahead: Monday T's signal is never applied before Monday T.

    Returns:
        Daily pd.Series of prob_up, NaN during warm-up period.
    """
    prices = df[price_col]

    # Step 1: Weekly resample — W-MON = week ending Monday, labeled on Monday
    weekly_prices = prices.resample('W-MON').last()
    log_prices_w = np.log(weekly_prices.values)

    # Step 2: Rolling OLS on weekly log prices
    slopes_short, r2_short = _fast_rolling_ols(log_prices_w, short_weeks)
    slopes_long,  r2_long  = _fast_rolling_ols(log_prices_w, long_weeks)

    # Steps 3+4: Annualize (×52) and blend TrendScore
    trend_score_w = (0.4 * slopes_short * 52.0 * r2_short +
                     0.6 * slopes_long  * 52.0 * r2_long)

    # Step 5: Z-score normalization on weekly series
    # min_periods: ~25% of zscore_weeks, minimum 8 weeks (mirrors daily min_periods=60)
    zscore_min = max(8, zscore_weeks // 4)
    ts_series_w = pd.Series(trend_score_w, index=weekly_prices.index)
    ts_mean_w   = ts_series_w.rolling(zscore_weeks, min_periods=zscore_min).mean()
    ts_std_w    = ts_series_w.rolling(zscore_weeks, min_periods=zscore_min).std().clip(lower=1e-8)
    ts_z_w      = ((ts_series_w - ts_mean_w) / ts_std_w).clip(-2.0, 2.0)

    # Step 6: Map to prob_up
    prob_up_w = pd.Series(
        0.5 - amplitude * np.tanh(ts_z_w.values),
        index=weekly_prices.index,
    )
    # Preserve NaN mask from warm-up period
    prob_up_w = prob_up_w.where(ts_z_w.notna(), other=np.nan)

    # Step 7: Forward-fill to daily index
    prob_up_daily = prob_up_w.reindex(df.index, method='ffill')

    return prob_up_daily


def run_evaluation(df_with_prob_up: pd.DataFrame, label: str) -> dict:
    """Run rolling window evaluation and return summary metrics."""
    results = evaluate_rolling_windows(
        df_with_prob_up,
        model=None,
        window_days=365,
        step_days=STEP_DAYS,
        verbose=False,
    )
    rw_pct      = calculate_recency_weighted_percentile(
        results['pct_strategy'].values, decay=DECAY
    )
    win_rate    = float((results['pct_strategy'] > results['pct_uniform']).mean())
    spd_strat   = float(results['spd_strategy'].mean())
    spd_uniform = float(results['spd_uniform'].mean())
    advantage   = spd_strat - spd_uniform
    n_windows   = len(results)

    print(f"  {label:10s}: RW% = {rw_pct:.2f}%  Win% = {win_rate:.2%}"
          f"  StratSPD = {spd_strat:,.1f}  Advantage = {advantage:+.1f}")

    return {
        'label':       label,
        'rw_pct':      rw_pct,
        'win_rate':    win_rate,
        'spd_strat':   spd_strat,
        'spd_uniform': spd_uniform,
        'advantage':   advantage,
        'n_windows':   n_windows,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("WEEKLY GRANULARITY ANALYSIS")
    print("=" * 72)
    print("""
Hypothesis: Weekly resampling reduces short-term BTC noise, aligns OLS
signal with 6-18 month cycle structure, and improves RW% vs daily baseline.

Decision rule (same as all prior experiments):
  Adopt if:  ΔRW > 0  AND  (win_rate stable OR SPD advantage improves)
  Reject if: ΔRW marginal (<0.25 pp) with degraded win_rate and SPD edge
""")

    np.random.seed(RANDOM_SEED)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading tournament data (step={STEP_DAYS}d, DECAY={DECAY})...")
    df = pd.read_parquet(DATA_URL)
    df = df.loc[BACKTEST_START:BACKTEST_END]
    print(f"  Daily range:  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Daily rows:   {len(df):,}")

    weekly_prices = df[PRICE_COL].resample('W-MON').last()
    print(f"  Weekly rows:  {len(weekly_prices):,}  "
          f"({weekly_prices.index[0].date()} → {weekly_prices.index[-1].date()})")

    # ── Build baseline (daily OLS 60/180/252) ─────────────────────────────────
    print("\nBuilding daily OLS baseline (short=60d, long=180d, z=252d)...")
    df_baseline = construct_features(df.copy())
    valid_base  = df_baseline['prob_up'].notna().sum()
    print(f"  Valid prob_up rows: {valid_base:,} / {len(df):,}  "
          f"(NaN warmup: {len(df)-valid_base} days ≈ {(len(df)-valid_base)//7}w)")

    # ── Evaluations ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ROLLING WINDOW EVALUATION")
    print("=" * 72)
    print(f"(step={STEP_DAYS}d, window=365d, DECAY={DECAY})\n")

    res_base = run_evaluation(df_baseline, label="Daily OLS")

    print()
    weekly_results = []
    for params in WINDOW_GRID:
        prob_up_w = build_weekly_ols_prob_up(
            df,
            short_weeks=params['short'],
            long_weeks=params['long'],
            zscore_weeks=params['zscore'],
        )
        valid_w = prob_up_w.notna().sum()

        df_variant          = df.copy()
        df_variant['prob_up'] = prob_up_w

        res = run_evaluation(df_variant, label=params['label'])
        res['short']      = params['short']
        res['long']       = params['long']
        res['zscore']     = params['zscore']
        res['valid_rows'] = valid_w
        weekly_results.append(res)

    # ── Parameter sweep table ─────────────────────────────────────────────────
    best = max(weekly_results, key=lambda r: r['rw_pct'])

    print("\n" + "=" * 72)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 72)
    print(f"\n  {'Variant':10s} | {'RW%':>6} | {'ΔRW':>7} | {'Win%':>6} | {'Adv(sats/$)':>12} | {'Valid days':>10}")
    print(f"  {'-'*66}")
    print(f"  {'Daily OLS':10s} | {res_base['rw_pct']:>5.2f}% | {'---':>7} | "
          f"{res_base['win_rate']:>5.2%} | {res_base['advantage']:>+11.1f} | "
          f"{int(df_baseline['prob_up'].notna().sum()):>10,}")
    for r in weekly_results:
        delta  = r['rw_pct'] - res_base['rw_pct']
        marker = " ★" if r['label'] == best['label'] else ""
        print(f"  {r['label']:10s} | {r['rw_pct']:>5.2f}% | {delta:>+6.2f}pp | "
              f"{r['win_rate']:>5.2%} | {r['advantage']:>+11.1f} | "
              f"{int(r['valid_rows']):>10,}{marker}")

    # ── Head-to-head: daily vs best weekly ────────────────────────────────────
    delta_rw  = best['rw_pct']  - res_base['rw_pct']
    delta_wr  = best['win_rate'] - res_base['win_rate']
    delta_spd = best['advantage'] - res_base['advantage']

    print("\n" + "=" * 72)
    print(f"HEAD-TO-HEAD: Daily Baseline vs Best Weekly ({best['label'].strip()})")
    print("=" * 72)
    print(f"""
  {'Metric':35} | {'Daily OLS':>16} | {'Best Weekly':>16} | {'Delta':>10}
  {'-'*87}
  {'RW Percentile (primary metric)':35} | {res_base['rw_pct']:>15.2f}% | {best['rw_pct']:>15.2f}% | {delta_rw:>+9.2f}pp
  {'Win Rate vs DCA':35} | {res_base['win_rate']:>15.2%} | {best['win_rate']:>15.2%} | {delta_wr:>+9.2%}
  {'Mean Strat SPD (sats/$)':35} | {res_base['spd_strat']:>16,.1f} | {best['spd_strat']:>16,.1f} | {best['spd_strat']-res_base['spd_strat']:>+9.1f}
  {'Mean DCA SPD (sats/$)':35} | {res_base['spd_uniform']:>16,.1f} | {best['spd_uniform']:>16,.1f} | {best['spd_uniform']-res_base['spd_uniform']:>+9.1f}
  {'Absolute SPD advantage (sats/$)':35} | {res_base['advantage']:>+16.1f} | {best['advantage']:>+16.1f} | {delta_spd:>+9.1f}
  {'vs Neutral baseline (41.94%)':35} | {res_base['rw_pct']-NEUTRAL_RW_PCT:>+15.2f}pp | {best['rw_pct']-NEUTRAL_RW_PCT:>+15.2f}pp | {'':>10}
  {'Windows evaluated':35} | {res_base['n_windows']:>16,} | {best['n_windows']:>16,} | {'':>10}
""")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)

    win_rate_ok = delta_wr  > -0.05   # within 5 pp degradation
    spd_ok      = delta_spd > -50     # within 50 sats/$ degradation
    rw_strong   = delta_rw  > 0.75
    rw_moderate = delta_rw  > 0.25
    rw_marginal = delta_rw  > 0.05

    if rw_strong and (win_rate_ok or spd_ok):
        category = "STRONG IMPROVEMENT"
        action   = "ADOPT — update construct_features to use weekly OLS"
    elif rw_moderate and (win_rate_ok or spd_ok):
        category = "MODERATE IMPROVEMENT"
        action   = "CONSIDER — validate with step=1 before adopting"
    elif rw_marginal:
        category = "MARGINAL IMPROVEMENT"
        action   = "DO NOT ADOPT — noise-level delta; daily OLS sufficient"
    elif delta_rw > -0.25:
        category = "NEUTRAL"
        action   = "DO NOT ADOPT — no improvement over daily baseline"
    else:
        category = "DEGRADATION"
        action   = "DO NOT ADOPT — weekly granularity underperforms daily"

    print(f"""
  Best weekly variant:  {best['label'].strip()}  (short={best['short']}w, long={best['long']}w, z={best['zscore']}w)
  ΔRW vs daily:         {delta_rw:+.2f} pp
  ΔWin rate:            {delta_wr:+.2%}
  ΔSPD advantage:       {delta_spd:+.1f} sats/$

  Category:       {category}
  Recommendation: {action}

  ── MECHANISM EXPLANATION ─────────────────────────────────────────────────

  Weekly resampling filters:
    - BTC's 24/7 weekend noise (thin liquidity, short-term reversals)
    - Intra-week oscillations that don't reflect the dominant trend
    - Short-term OLS slope instability from daily price jumps

  The core hypothesis: BTC's major cycles (accumulation, mark-up, distribution,
  mark-down) operate on 26-78 week timescales. A 12w/26w weekly OLS captures
  these phases more cleanly than 60d/180d daily OLS.

  Counter-hypothesis (null result):
    The daily 252d rolling z-score normalization already suppresses noise by
    standardizing TrendScore against its own rolling volatility. Weekly
    resampling may be redundant once z-score normalization is applied.

    Daily approximate weekly equivalents:
      short_window=60d  ≈ {60//7}w   (weekly primary: {WINDOW_GRID[1]['short']}w)
      long_window=180d  ≈ {180//7}w  (weekly primary: {WINDOW_GRID[1]['long']}w)
      z_window=252d     ≈ {252//7}w  (weekly primary: {WINDOW_GRID[1]['zscore']}w)

  ── NARRATIVE VALUE FOR FINAL REPORT ──────────────────────────────────────

  This experiment probes whether temporal granularity of OLS computation
  is a meaningful design choice after z-score normalization.

  Either outcome provides a clean narrative arc:
    ΔRW > 1pp: "BTC cycle structure is better captured at weekly
      timescale — the 6-18 month macro cycles dominate the sats
      accumulation signal, and weekly OLS aligns naturally with them."
    |ΔRW| < 0.5pp: "Temporal granularity is not a significant factor
      once z-score normalization is applied. The 252d rolling window
      already captures cycle dynamics at daily resolution — confirming
      that our post-midterm pivot to z-score normalization was the
      structural insight, not the OLS window size."
""")
    print("=" * 72)

    return {
        'baseline':       res_base,
        'weekly_results': weekly_results,
        'best_weekly':    best,
        'delta_rw':       delta_rw,
    }


if __name__ == "__main__":
    results = main()
