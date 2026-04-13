"""
Tournament Mode for Visual Trading System (VTS)

Adapter layer to convert VTS continuous CNN-based allocator to
tournament-compliant normalized daily allocation weights.

Author: VTS Team
Date: 2026-01-21
Tournament: Trilemma x Strategy Stacking Sats Tournament
"""

__version__ = "1.0.0"

from .scoring import (
    calculate_spd_for_window,
    calculate_recency_weighted_percentile,
    calculate_win_rate,
    MIN_WEIGHT,
    TOLERANCE
)

from .weights import (
    compute_weights,
    enforce_constraints
)

from .evaluator import (
    evaluate_rolling_windows,
    evaluate_single_window,
    summarize_results,
    iter_rolling_windows
)

# Tournament-required endpoints (lightweight, no model loading)
from .features_simplified import construct_features

__all__ = [
    # Tournament-required endpoints
    'construct_features',
    'compute_weights',
    # Scoring functions
    'calculate_spd_for_window',
    'calculate_recency_weighted_percentile',
    'calculate_win_rate',
    # Weight utilities
    'enforce_constraints',
    # Evaluation framework
    'evaluate_rolling_windows',
    'evaluate_single_window',
    'summarize_results',
    'iter_rolling_windows',
    # Constants
    'MIN_WEIGHT',
    'TOLERANCE'
]
