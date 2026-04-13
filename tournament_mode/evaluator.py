"""
tournament_mode.evaluator

Rolling-window tournament evaluator.

Stitches together:
    construct_features -> compute_weights -> scoring
over rolling 12-month windows.

Design goals:
- deterministic and index-aligned
- causality preserved (relies on features/weights modules to be causal)
- can run with precomputed prob_up (skip CNN) or by calling construct_features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .features import construct_features
    _FEATURES_BACKEND = "CNN"
except (ImportError, ModuleNotFoundError):
    import warnings
    warnings.warn(
        "torch not available — using features_simplified (OLS) backend, not CNN. "
        "Set features_backend=SIMPLIFIED in run metadata.",
        ImportWarning, stacklevel=2
    )
    from .features_simplified import construct_features  # fallback when torch absent
    _FEATURES_BACKEND = "SIMPLIFIED"
from .weights import compute_weights, uniform_weights

# Use our existing scoring functions
from .scoring import (
    calculate_spd_for_window,
    calculate_recency_weighted_percentile,
    calculate_win_rate,
)


@dataclass(frozen=True)
class WindowResult:
    """Results for a single 12-month window."""
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    spd_strategy: float
    spd_uniform: float
    pct_strategy: float
    pct_uniform: float


def _validate_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare input DataFrame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input df must have a DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError("Input df index contains duplicates")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if "PriceUSD_coinmetrics" not in df.columns:
        raise ValueError("Input df must include 'PriceUSD_coinmetrics' column")
    return df


def iter_rolling_windows(
    df: pd.DataFrame,
    window_days: int = 365,
    step_days: int = 1,
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """
    Yield (window_idx, df_window) for rolling windows by row count.

    This matches the typical "daily bars, 12-month window, step=1 day" pattern
    and the ~3,075 window expectation for 2016-2025 daily series.

    Args:
        df: Input DataFrame with DatetimeIndex
        window_days: Window size in days (default 365)
        step_days: Step size in days (default 1)

    Yields:
        Tuple of (window_idx, df_window)
    """
    if window_days <= 1:
        raise ValueError("window_days must be > 1")
    if step_days <= 0:
        raise ValueError("step_days must be >= 1")

    n = len(df)
    if n < window_days:
        return

    idx = 0
    for end_pos in range(window_days - 1, n, step_days):
        start_pos = end_pos - (window_days - 1)
        yield idx, df.iloc[start_pos : end_pos + 1]
        idx += 1


def evaluate_single_window(
    df_window: pd.DataFrame,
    *,
    model=None,
    features_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    weights_fn: Callable[[pd.DataFrame], pd.Series] = compute_weights,
) -> WindowResult:
    """
    Run the full pipeline on a single window.

    If df_window already contains 'prob_up', we will use it directly and skip CNN.
    Otherwise, we call features_fn (construct_features).

    Args:
        df_window: 12-month window DataFrame
        model: Optional pre-trained CNN model
        features_fn: Feature generation function (default: construct_features)
        weights_fn: Weight computation function (default: compute_weights)

    Returns:
        WindowResult with SPD and percentile metrics
    """
    df_window = _validate_input_df(df_window)

    # Build features
    if "prob_up" in df_window.columns:
        # Already has features, use directly
        feats = df_window.copy()
        feats["prob_up"] = feats["prob_up"].astype(float)
    else:
        # Generate features using CNN
        if features_fn is None:
            features_fn = construct_features

        feats = features_fn(df_window) if model is None else features_fn(df_window, model)

        # Ensure price column exists in features
        if "PriceUSD_coinmetrics" not in feats.columns:
            feats = feats.copy()
            feats["PriceUSD_coinmetrics"] = df_window["PriceUSD_coinmetrics"]

    # Compute weights
    w = weights_fn(feats)
    if not isinstance(w, pd.Series):
        w = pd.Series(w, index=feats.index, dtype=float)
    w = w.astype(float)

    # Extract prices
    prices = feats["PriceUSD_coinmetrics"].astype(float)

    # Calculate SPD metrics using our comprehensive function
    result_strategy = calculate_spd_for_window(w, prices)

    # Uniform baseline
    w_u = uniform_weights(feats)
    result_uniform = calculate_spd_for_window(w_u, prices)

    return WindowResult(
        start=prices.index[0],
        end=prices.index[-1],
        n_days=len(prices),
        spd_strategy=result_strategy['dynamic_spd'],
        spd_uniform=result_uniform['dynamic_spd'],
        pct_strategy=result_strategy['dynamic_pct'],
        pct_uniform=result_uniform['dynamic_pct'],
    )


def evaluate_rolling_windows(
    df: pd.DataFrame,
    *,
    model=None,
    window_days: int = 365,
    step_days: int = 1,
    features_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    weights_fn: Callable[[pd.DataFrame], pd.Series] = compute_weights,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all rolling windows. Returns tidy per-window results.

    Args:
        df: Input DataFrame with DatetimeIndex and PriceUSD_coinmetrics column
        model: Optional pre-trained CNN model
        window_days: Window size (default 365)
        step_days: Step size (default 1)
        features_fn: Feature generation function
        weights_fn: Weight computation function
        verbose: Print progress (default True)

    Returns:
        DataFrame with one row per window containing metrics
    """
    df = _validate_input_df(df)

    rows = []
    total_windows = (len(df) - window_days) // step_days + 1

    for window_idx, df_w in iter_rolling_windows(df, window_days=window_days, step_days=step_days):
        r = evaluate_single_window(
            df_w,
            model=model,
            features_fn=features_fn,
            weights_fn=weights_fn
        )

        rows.append({
            "window_idx": window_idx,
            "start": r.start,
            "end": r.end,
            "n_days": r.n_days,
            "spd_strategy": r.spd_strategy,
            "spd_uniform": r.spd_uniform,
            "pct_strategy": r.pct_strategy,
            "pct_uniform": r.pct_uniform,
        })

        # Progress indicator
        if verbose and (window_idx + 1) % 100 == 0:
            print(f"  Evaluated {window_idx + 1}/{total_windows} windows...")

    out = pd.DataFrame(rows)
    if not out.empty:
        out["start"] = pd.to_datetime(out["start"])
        out["end"] = pd.to_datetime(out["end"])

    if verbose:
        print(f"\n✅ Completed {len(out)} rolling windows")

    return out


def summarize_results(results: pd.DataFrame, *, decay: float = 0.9) -> Dict[str, float]:
    """
    Compute tournament-style aggregates from per-window results.

    Args:
        results: DataFrame from evaluate_rolling_windows()
        decay: Recency weighting decay parameter (default 0.9)

    Returns:
        Dictionary with tournament metrics:
            - n_windows: Number of windows evaluated
            - win_rate: Fraction of windows beating DCA
            - rw_spd_percentile: Recency-weighted SPD percentile (PRIMARY)
            - mean_spd_percentile: Simple average percentile
    """
    if results.empty:
        return {
            "n_windows": 0.0,
            "win_rate": float("nan"),
            "rw_spd_percentile": float("nan"),
            "mean_spd_percentile": float("nan"),
        }

    # Extract percentiles (oldest first, newest last - chronological order)
    percentiles_strategy = results["pct_strategy"].values
    percentiles_uniform = results["pct_uniform"].values

    # Recency-weighted percentile (PRIMARY metric)
    rw_pct = float(calculate_recency_weighted_percentile(
        percentiles_strategy,
        decay=decay
    ))

    # Win rate (SECONDARY metric)
    win_rate = float(calculate_win_rate(percentiles_strategy, percentiles_uniform))

    # Simple average for comparison
    mean_pct = float(percentiles_strategy.mean())

    return {
        "n_windows": float(len(results)),
        "win_rate": win_rate,
        "rw_spd_percentile": rw_pct,
        "mean_spd_percentile": mean_pct,
    }


def _main() -> int:
    """Command-line interface for evaluator."""
    import argparse

    p = argparse.ArgumentParser(description="Run rolling-window tournament evaluation")
    p.add_argument("--csv", type=str, required=True, help="CSV with Date column and at least price")
    p.add_argument("--date-col", type=str, default="Date", help="Date column name")
    p.add_argument("--price-col", type=str, default="PriceUSD_coinmetrics", help="Price column name")
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--step-days", type=int, default=1)
    p.add_argument("--decay", type=float, default=0.9)
    args = p.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.set_index(args.date_col).sort_index()

    if args.price_col != "PriceUSD_coinmetrics":
        df = df.rename(columns={args.price_col: "PriceUSD_coinmetrics"})

    # Evaluate
    print(f"\nEvaluating rolling windows...")
    print(f"  Window size: {args.window_days} days")
    print(f"  Step size: {args.step_days} days")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    results = evaluate_rolling_windows(
        df,
        window_days=args.window_days,
        step_days=args.step_days
    )

    summary = summarize_results(results, decay=args.decay)

    # Print results
    print("\n" + "=" * 70)
    print("TOURNAMENT EVALUATION SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        if k == "win_rate":
            print(f"{k}: {v:.2%}")
        elif "percentile" in k:
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.0f}")

    print("\nPer-window results (first 5):")
    print(results.head(5).to_string(index=False))

    print("\nPer-window results (last 5):")
    print(results.tail(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
