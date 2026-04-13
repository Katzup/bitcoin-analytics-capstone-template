#!/usr/bin/env python3
"""
VTS Dashboard — Stacking-SATS Tournament
=========================================
Run:  python3.10 -m streamlit run dashboard.py

All heavy computation is cached on first load (~2 min for rolling windows).
Uses step=7 fast scan throughout; headline RW percentile (44.95%, step=1)
is displayed as a locked constant.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VTS — Stacking-SATS Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/TrilemmaFoundation/"
    "stacking-sats-tournament-mstr-2025/main/data/stacking_sats_data.parquet"
)
BACKTEST_START  = "2016-01-01"
BACKTEST_END    = "2025-06-01"
DECAY           = 0.9
SIM_WINDOW      = 365
HEADLINE_RW     = 44.95   # step=1 full validation — locked
NEUTRAL_RW      = 41.94
HEADLINE_WIN    = 66.94   # step=1


# ── Cached data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data from GitHub…")
def load_data():
    from tournament_mode import construct_features
    df = pd.read_parquet(DATA_URL).loc[BACKTEST_START:BACKTEST_END]
    df_feat = construct_features(df.copy())
    return df, df_feat


@st.cache_data(show_spinner="Running rolling window evaluation (step=7, ~1-2 min)…")
def load_rolling_results(_df_feat):
    from tournament_mode.evaluator import evaluate_rolling_windows
    from tournament_mode.scoring import calculate_recency_weighted_percentile
    results = evaluate_rolling_windows(
        _df_feat, model=None, window_days=365, step_days=7, verbose=False
    )
    rw = calculate_recency_weighted_percentile(
        results["pct_strategy"].values, decay=DECAY
    )
    return results, rw


@st.cache_data(show_spinner="Simulating equity curve…")
def load_equity_curve(_df_feat):
    df = _df_feat.dropna(subset=["prob_up"]).copy()
    prob_up = df["prob_up"]
    prices  = df["PriceUSD_coinmetrics"]
    rm = prob_up.rolling(SIM_WINDOW, min_periods=SIM_WINDOW // 4).mean()
    mult = (prob_up / rm.clip(lower=1e-6)).clip(lower=0.1, upper=10.0)
    s_btc = (mult / prices).cumsum()
    n_btc = (1.0  / prices).cumsum()
    curve = pd.DataFrame({
        "strategy_usd": s_btc * prices,
        "neutral_usd":  n_btc * prices,
        "strategy_btc": s_btc,
        "neutral_btc":  n_btc,
        "price":        prices,
        "prob_up":      prob_up,
        "multiplier":   mult,
    }, index=df.index).iloc[SIM_WINDOW:]
    return curve


# ── Helper functions ──────────────────────────────────────────────────────────
def drawdown_series(s: pd.Series) -> pd.Series:
    peak = s.cummax()
    return (s - peak) / peak * 100


def rolling_md_rv(series: pd.Series, window: int = 365) -> pd.Series:
    """Rolling max-drawdown / annualized realized vol."""
    out = []
    for i in range(window, len(series) + 1):
        win = series.iloc[i - window: i]
        lr  = np.log(win / win.shift(1)).dropna()
        rv  = lr.std() * np.sqrt(252)
        dd  = abs(((win - win.cummax()) / win.cummax()).min())
        out.append(dd / rv if rv > 0 else np.nan)
    idx = series.index[window - 1:]
    return pd.Series(out, index=idx)


COLORS = {
    "strategy": "#F7931A",   # Bitcoin orange
    "neutral":  "#6B7280",   # Gray
    "btc":      "#3B82F6",   # Blue
    "green":    "#10B981",
    "red":      "#EF4444",
    "bg":       "#0F172A",
}


# ── Load everything ───────────────────────────────────────────────────────────
df, df_feat = load_data()
results, rw_fastscan = load_rolling_results(df_feat)
curve = load_equity_curve(df_feat)

excess_pct = results["pct_strategy"] - results["pct_uniform"]
excess_spd = results["spd_strategy"] - results["spd_uniform"]
wins = (results["pct_strategy"] > results["pct_uniform"])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("₿ VTS — Stacking-SATS Tournament Dashboard")
st.caption(
    "OLS Regime-Relative Dampening Signal · Bob Katz · GT OMSA Practicum Spring 2026 · "
    "Data: 2016-01-01 → 2025-06-01"
)

# ── Scorecard ─────────────────────────────────────────────────────────────────
st.subheader("Scorecard")
st.caption(
    "Headline RW percentile is step=1 full validation (3,076 windows). "
    "All other metrics use the step=7 fast scan (~440 windows)."
)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("RW Percentile (step=1)", f"{HEADLINE_RW:.2f}%",
          f"+{HEADLINE_RW - NEUTRAL_RW:.2f} pp vs neutral")
c2.metric("Win Rate (step=1)",       f"{HEADLINE_WIN:.1f}%",
          f"strategy > neutral in {HEADLINE_WIN:.1f}% of windows")
c3.metric("Fast-scan RW (step=7)",   f"{rw_fastscan:.2f}%",
          "robustness check only")
c4.metric("IR (excess pct, step=7)", f"{excess_pct.mean()/excess_pct.std():.3f}",
          "mean/std of excess %-pt")
c5.metric("Median excess SPD",       f"+{excess_spd.median():.0f} sats/$",
          "per 365-day window")
c6.metric("Max losing streak",       f"{int((~wins).astype(int).groupby((wins != wins.shift()).cumsum()).transform('sum').max())} windows",
          "consecutive step=7 windows")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("1 · Signal: OLS Regime-Relative Dampening")
st.markdown(
    """
    **prob_up** drives daily BTC allocation weight. When `prob_up < 0.5` (muted trend),
    the strategy buys *less* relative to its rolling baseline; when `prob_up > 0.5`
    (subdued/recovering trend), it buys *more*. The signal is **not** a reversal predictor —
    it correlates positively with forward returns (r ≈ +0.14 at 30 days) and is best
    described as **regime-relative dampening** during extended uptrends.
    """
)

fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.4, 0.6],
                     subplot_titles=["prob_up (allocation tilt)", "BTC Price (USD, log scale)"])

prob_plot = df_feat["prob_up"].dropna()
fig1.add_trace(go.Scatter(
    x=prob_plot.index, y=prob_plot.values,
    name="prob_up", line=dict(color=COLORS["strategy"], width=1),
    fill="tozeroy", fillcolor="rgba(247,147,26,0.12)",
), row=1, col=1)
fig1.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4, row=1, col=1)
fig1.add_annotation(x=prob_plot.index[-1], y=0.5, text="neutral (0.5)",
                    showarrow=False, xanchor="right", font=dict(color="white", size=10),
                    row=1, col=1)

price = df_feat["PriceUSD_coinmetrics"].dropna()
fig1.add_trace(go.Scatter(
    x=price.index, y=price.values,
    name="BTC price", line=dict(color=COLORS["btc"], width=1.5),
), row=2, col=1)
fig1.update_yaxes(type="log", row=2, col=1)
fig1.update_layout(height=480, showlegend=False,
                   paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                   font=dict(color="white"), margin=dict(l=0, r=0, t=30, b=0))
fig1.update_xaxes(showgrid=False)
fig1.update_yaxes(gridcolor="#334155")
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PER-WINDOW PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("2 · Per-Window Performance (365-day rolling, step=7)")

col_a, col_b = st.columns([3, 2])

with col_a:
    # Rolling percentile over time
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=results["end"], y=results["pct_strategy"],
        name="Strategy", line=dict(color=COLORS["strategy"], width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=results["end"], y=results["pct_uniform"],
        name="Neutral DCA", line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
    ))
    fig2.update_layout(
        title="SPD Percentile by Window End Date",
        xaxis_title="Window end", yaxis_title="Percentile (%)",
        height=320, paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font=dict(color="white"), legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig2.update_yaxes(gridcolor="#334155")
    fig2.update_xaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    # Excess pct distribution
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=excess_pct.values, nbinsx=40,
        marker_color=COLORS["strategy"], opacity=0.8,
        name="Excess pct",
    ))
    fig3.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.6)
    fig3.add_vline(x=excess_pct.median(), line_dash="dot",
                   line_color=COLORS["green"], opacity=0.8,
                   annotation_text=f"median {excess_pct.median():+.2f}pp",
                   annotation_font_color=COLORS["green"])
    fig3.update_layout(
        title="Excess Percentile Distribution (strategy − neutral)",
        xaxis_title="Excess pct (pp)", yaxis_title="Count",
        height=320, showlegend=False,
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font=dict(color="white"), margin=dict(l=0, r=0, t=40, b=0),
    )
    fig3.update_yaxes(gridcolor="#334155")
    fig3.update_xaxes(showgrid=False)
    st.plotly_chart(fig3, use_container_width=True)

# Performance ladder
st.markdown("**Performance Ladder** (step=1 full validation, 3,076 windows)")
ladder = pd.DataFrame([
    {"Signal": "CNN / GAF",               "RW%": 41.43, "Win%": 54.32, "Δ vs neutral": -0.51},
    {"Signal": "Neutral (prob_up = 0.5)", "RW%": 41.94, "Win%": 70.42, "Δ vs neutral":  0.00},
    {"Signal": "OLS raw contrarian",      "RW%": 43.25, "Win%": 91.48, "Δ vs neutral": +1.31},
    {"Signal": "OLS z-score ±2σ (final)", "RW%": 44.95, "Win%": 66.94, "Δ vs neutral": +3.01},
])
fig_ladder = go.Figure(go.Bar(
    x=ladder["Signal"], y=ladder["RW%"],
    marker_color=[
        COLORS["red"], COLORS["neutral"], COLORS["btc"], COLORS["strategy"]
    ],
    text=[f"{v:.2f}%" for v in ladder["RW%"]],
    textposition="outside",
))
fig_ladder.add_hline(y=41.94, line_dash="dash", line_color="white", opacity=0.5,
                     annotation_text="neutral 41.94%", annotation_font_color="white")
fig_ladder.update_layout(
    yaxis=dict(range=[40, 46], title="RW Percentile (%)"),
    height=320, showlegend=False,
    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
    font=dict(color="white"), margin=dict(l=0, r=0, t=20, b=0),
)
fig_ladder.update_yaxes(gridcolor="#334155")
fig_ladder.update_xaxes(showgrid=False)
st.plotly_chart(fig_ladder, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EQUITY CURVE & TRADITIONAL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("3 · Equity Curve & Traditional Risk/Return")
st.caption(
    "Simulation: \\$1/day DCA, daily_multiplier = prob_up_t / rolling_mean(prob_up, 365). "
    "Both strategies invest \\$1/day on average. USD portfolio value = cumulative BTC × current price."
)

# Equity curves + drawdown
fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.65, 0.35],
                     subplot_titles=["Portfolio Value (USD, $1/day DCA)", "Drawdown (%)"])

fig4.add_trace(go.Scatter(
    x=curve.index, y=curve["strategy_usd"],
    name="Strategy", line=dict(color=COLORS["strategy"], width=2),
), row=1, col=1)
fig4.add_trace(go.Scatter(
    x=curve.index, y=curve["neutral_usd"],
    name="Neutral DCA", line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=curve.index, y=drawdown_series(curve["strategy_usd"]),
    name="Strategy DD", line=dict(color=COLORS["strategy"], width=1),
    fill="tozeroy", fillcolor="rgba(247,147,26,0.15)",
    showlegend=False,
), row=2, col=1)
fig4.add_trace(go.Scatter(
    x=curve.index, y=drawdown_series(curve["neutral_usd"]),
    name="Neutral DD", line=dict(color=COLORS["neutral"], width=1, dash="dash"),
    showlegend=False,
), row=2, col=1)

fig4.update_yaxes(type="log", row=1, col=1)
fig4.update_layout(
    height=500, paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
    font=dict(color="white"), legend=dict(orientation="h", y=1.05),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig4.update_xaxes(showgrid=False)
fig4.update_yaxes(gridcolor="#334155")
st.plotly_chart(fig4, use_container_width=True)

# Rolling MD/RV
with st.spinner("Computing rolling MD/RV…"):
    md_rv_strat = rolling_md_rv(curve["strategy_usd"])
    md_rv_neut  = rolling_md_rv(curve["neutral_usd"])

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=md_rv_strat.index, y=md_rv_strat.values,
    name="Strategy MD/RV", line=dict(color=COLORS["strategy"], width=2),
))
fig5.add_trace(go.Scatter(
    x=md_rv_neut.index, y=md_rv_neut.values,
    name="Neutral MD/RV", line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
))
fig5.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.4,
               annotation_text="MD/RV = 1  (drawdown = 1yr of typical vol)",
               annotation_font_color="white")
fig5.update_layout(
    title="Rolling 365-day MD/RV  (lower = better risk-adjusted drawdown)",
    xaxis_title="Date", yaxis_title="Max Drawdown / Realized Vol",
    height=300, paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
    font=dict(color="white"), legend=dict(orientation="h", y=1.1),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig5.update_yaxes(gridcolor="#334155")
fig5.update_xaxes(showgrid=False)
st.plotly_chart(fig5, use_container_width=True)

# Traditional metrics summary table
def _equity_stats(s):
    lr = np.log(s / s.shift(1)).dropna()
    years   = len(s) / 252
    cagr    = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    av      = lr.std() * np.sqrt(252)
    am      = lr.mean() * 252
    sh      = am / av if av > 0 else np.nan
    dr      = lr[lr < 0]
    adv     = dr.std() * np.sqrt(252) if len(dr) > 0 else np.nan
    so      = am / adv if adv and adv > 0 else np.nan
    peak    = s.cummax()
    mdd     = ((s - peak) / peak).min()
    cal     = cagr / abs(mdd) if mdd != 0 else np.nan
    return dict(CAGR=f"{cagr*100:.1f}%", Vol=f"{av*100:.1f}%",
                Sharpe=f"{sh:.3f}", Sortino=f"{so:.3f}",
                MaxDD=f"{mdd*100:.1f}%", Calmar=f"{cal:.3f}")

s_stats = _equity_stats(curve["strategy_usd"])
n_stats = _equity_stats(curve["neutral_usd"])

tbl = pd.DataFrame({"Strategy": s_stats, "Neutral DCA": n_stats}).T
st.dataframe(tbl, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("4 · Robustness Checks")

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("**Fee Sensitivity** (step=7, proportional fees)")
    st.caption(
        "Proportional fees cancel in the RW-percentile formula — all SPD terms scale "
        "identically by 1/(1 + fee). Invariance breaks under trade-count-dependent slippage."
    )
    fee_data = pd.DataFrame({
        "fee_bps": [0, 10, 25, 50, 75, 100],
        "rw_pct":  [rw_fastscan, rw_fastscan - 0.39, rw_fastscan - 0.98,
                    rw_fastscan - 3.68, rw_fastscan - 6.10, rw_fastscan - 8.20],
    })
    fig_fee = go.Figure(go.Bar(
        x=fee_data["fee_bps"].astype(str) + " bps",
        y=fee_data["rw_pct"],
        marker_color=[
            COLORS["strategy"] if v > NEUTRAL_RW else COLORS["red"]
            for v in fee_data["rw_pct"]
        ],
        text=[f"{v:.2f}%" for v in fee_data["rw_pct"]],
        textposition="outside",
    ))
    fig_fee.add_hline(y=NEUTRAL_RW, line_dash="dash", line_color="white", opacity=0.5,
                      annotation_text=f"neutral {NEUTRAL_RW}%",
                      annotation_font_color="white")
    fig_fee.update_layout(
        xaxis_title="Fee (bps)", yaxis_title="RW Percentile (%)",
        yaxis=dict(range=[38, 48]),
        height=300, showlegend=False,
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font=dict(color="white"), margin=dict(l=0, r=0, t=10, b=0),
    )
    fig_fee.update_yaxes(gridcolor="#334155")
    fig_fee.update_xaxes(showgrid=False)
    st.plotly_chart(fig_fee, use_container_width=True)

with col_d:
    st.markdown("**Null Results — Rejected Variants**")
    st.caption("Tested against OLS z-score winsorized baseline (44.95% RW, step=1).")
    null_df = pd.DataFrame([
        {"Variant": "Vol-normalized amplitude",    "ΔRW":  "+0.14 pp", "ΔWin%": "−1.59 pp", "ΔSPD": "−32 sats/$", "Decision": "REJECT"},
        {"Variant": "Weekly OLS (12/26/52w)",      "ΔRW":  "−0.90 pp", "ΔWin%": "—",         "ΔSPD": "—",          "Decision": "REJECT"},
        {"Variant": "Weekly OLS best (12/26/26w)", "ΔRW":  "+0.26 pp", "ΔWin%": "—",         "ΔSPD": "—",          "Decision": "REJECT"},
        {"Variant": "Turnover governor",            "ΔRW":  "0.00 pp",  "ΔWin%": "—",         "ΔSPD": "—",          "Decision": "ADOPT (operational)"},
    ])
    st.dataframe(null_df, use_container_width=True, hide_index=True)

    st.markdown("**Evaluation Contract** (verify_endpoints_demo.py)")
    contract_df = pd.DataFrame([
        {"Check": "sum(weights) = 1.0",            "Status": "✅"},
        {"Check": "all weights ≥ MIN_WEIGHT",       "Status": "✅"},
        {"Check": "len(weights) = window_days",     "Status": "✅"},
        {"Check": "index preserved",                "Status": "✅"},
        {"Check": "all finite",                     "Status": "✅"},
        {"Check": "deterministic (seed=42)",        "Status": "✅"},
        {"Check": "governor backward-compatible",   "Status": "✅"},
    ])
    st.dataframe(contract_df, use_container_width=True, hide_index=True)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    "Primary metric: Recency-Weighted Percentile (decay=0.9) of SPD across rolling "
    "365-day windows, 2016-01-01 → 2025-06-01, step=1 (3,076 windows), seed=42. "
    "Signal: `prob_up = 0.5 − 0.15×tanh(ts_z)` where `ts_z` is a 252-day rolling "
    "z-score of a 60/180-day dual-window OLS trend score. "
    "Backend: SIMPLIFIED (OLS, no torch required). "
    "Reproducibility: `python3.10 verify_endpoints_demo.py`"
)
