"""
Charting module for Trade Buddy (v3 — clean & readable).

Principles:
  - Main chart shows ONLY: candlesticks + EMA(9) + EMA(21) + buy/sell markers
  - Extra overlays (BB, VWAP, S/R) are optional and toggled in the UI
  - MACD lives in its own separate, smaller chart — not crammed below candles
  - Volume is a simple, slim subplot
  - Neutral palette, no label clutter on the chart itself
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, List

_GREEN  = "#26a69a"
_RED    = "#ef5350"
_EMA9   = "#29b6f6"   # blue
_EMA21  = "#ce93d8"   # light purple
_BB_COL = "rgba(255, 235, 59, 0.18)"
_GRID   = "#1f2638"


def _x(df: pd.DataFrame):
    """Shared x-axis values — plain string timestamps."""
    return df["timestamp"].astype(str)


def create_price_chart(
    df: pd.DataFrame,
    show_bb: bool = False,
    show_vwap: bool = False,
    show_signals: bool = True,
    show_sr: bool = True,
    support_levels: Optional[List[float]] = None,
    resistance_levels: Optional[List[float]] = None,
    highlight_ts: Optional[str] = None,
    title: str = "NIFTY 50 · 1-min",
) -> go.Figure:
    """
    2-panel chart: candlesticks (+ optional overlays) on top, slim volume bar below.
    Clean by default — only EMAs and buy/sell markers are always shown.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )
    x = _x(df)

    # ── Candlesticks ──────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="NIFTY 50",
        increasing=dict(line=dict(color=_GREEN, width=1), fillcolor=_GREEN),
        decreasing=dict(line=dict(color=_RED,   width=1), fillcolor=_RED),
        whiskerwidth=0.5,
    ), row=1, col=1)

    # ── EMA 9 (always on) ─────────────────────────────────
    if "ema_9" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["ema_9"], name="EMA 9",
            line=dict(color=_EMA9, width=1.4), opacity=0.9,
        ), row=1, col=1)

    # ── EMA 21 (always on) ────────────────────────────────
    if "ema_21" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["ema_21"], name="EMA 21",
            line=dict(color=_EMA21, width=1.4), opacity=0.9,
        ), row=1, col=1)

    # ── Optional: Bollinger Bands ─────────────────────────
    if show_bb and "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["bb_upper"], name="BB Upper",
            line=dict(color="rgba(255,235,59,0.6)", width=1, dash="dot"),
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["bb_lower"], name="BB Lower",
            line=dict(color="rgba(255,235,59,0.6)", width=1, dash="dot"),
            fill="tonexty", fillcolor=_BB_COL, showlegend=True,
        ), row=1, col=1)

    # ── Optional: VWAP ───────────────────────────────────
    if show_vwap and "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["vwap"], name="VWAP",
            line=dict(color="rgba(255,167,38,0.85)", width=1.3),
        ), row=1, col=1)

    # ── Optional: Support & Resistance (max 3 each, thin dashes) ─
    if show_sr:
        for lvl in (support_levels or [])[:3]:
            fig.add_hline(y=lvl, line_dash="dash",
                          line=dict(color="rgba(38,166,154,0.55)", width=1),
                          row=1, col=1,
                          annotation_text=f"S {lvl:,.0f}",
                          annotation_position="right",
                          annotation_font=dict(size=9, color="rgba(38,166,154,0.9)"))
        for lvl in (resistance_levels or [])[:3]:
            fig.add_hline(y=lvl, line_dash="dash",
                          line=dict(color="rgba(239,83,80,0.55)", width=1),
                          row=1, col=1,
                          annotation_text=f"R {lvl:,.0f}",
                          annotation_position="right",
                          annotation_font=dict(size=9, color="rgba(239,83,80,0.9)"))

    # ── Buy / Sell signal markers ─────────────────────────
    if show_signals and "signal_type" in df.columns:
        buy_m  = df["signal_type"] == "buy"
        sell_m = df["signal_type"] == "sell"

        if buy_m.any():
            fig.add_trace(go.Scatter(
                x=x[buy_m], y=df["low"][buy_m] * 0.9992,
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=9, color="#00e676",
                            line=dict(width=0.8, color="white")),
                hovertemplate="BUY<br>%{x}",
            ), row=1, col=1)

        if sell_m.any():
            fig.add_trace(go.Scatter(
                x=x[sell_m], y=df["high"][sell_m] * 1.0008,
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=9, color="#ff1744",
                            line=dict(width=0.8, color="white")),
                hovertemplate="SELL<br>%{x}",
            ), row=1, col=1)

    # ── Highlighted evaluation candle ────────────────────
    if highlight_ts:
        h = df[df["timestamp"].astype(str) == highlight_ts]
        if not h.empty:
            fig.add_trace(go.Scatter(
                x=[highlight_ts], y=[h.iloc[0]["close"]],
                mode="markers", name="Evaluated",
                marker=dict(symbol="star", size=12, color="#ffeb3b",
                            line=dict(width=1, color="#333")),
                showlegend=False,
            ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────
    vol_colors = [
        _GREEN if float(df.iloc[i]["close"]) >= float(df.iloc[i]["open"]) else _RED
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=x, y=df["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.65, showlegend=False,
    ), row=2, col=1)

    if "vol_ma_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["vol_ma_20"], name="Vol MA20",
            line=dict(color="rgba(255,167,38,0.75)", width=1.1, dash="dot"),
            showlegend=False,
        ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=55, r=80, t=35, b=20),
        xaxis_rangeslider_visible=False,
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        height=580,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#90a4ae"), x=0.01),
    )
    fig.update_yaxes(title_text="₹ Points", row=1, col=1,
                     gridcolor=_GRID, tickfont=dict(size=10), fixedrange=False)
    fig.update_yaxes(title_text="Vol", row=2, col=1,
                     gridcolor=_GRID, tickfont=dict(size=9), fixedrange=False)
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9), fixedrange=False)

    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """
    Standalone MACD chart — shown in an expandable section, not crammed into the main chart.
    """
    if "macd" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    x   = _x(df)

    hist_colors = [
        _GREEN if float(v) >= 0 else _RED
        for v in df["macd_histogram"].fillna(0)
    ]

    fig.add_trace(go.Bar(
        x=x, y=df["macd_histogram"], name="Histogram",
        marker_color=hist_colors, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["macd"], name="MACD",
        line=dict(color=_EMA9, width=1.4),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["macd_signal"], name="Signal",
        line=dict(color="#ffa726", width=1.2, dash="dash"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=55, r=80, t=28, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        height=200,
        title=dict(text="<b>MACD (12, 26, 9)</b>",
                   font=dict(size=12, color="#90a4ae"), x=0.01),
        yaxis=dict(gridcolor=_GRID, tickfont=dict(size=9)),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(size=9)),
    )
    return fig


def create_stoch_chart(df: pd.DataFrame) -> go.Figure:
    """Standalone Stochastic Oscillator chart."""
    if "stoch_k" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    x   = _x(df)

    fig.add_trace(go.Scatter(x=x, y=df["stoch_k"], name="%K",
                             line=dict(color=_EMA9, width=1.3)))
    fig.add_trace(go.Scatter(x=x, y=df["stoch_d"], name="%D",
                             line=dict(color="#ffa726", width=1.1, dash="dash")))

    # Overbought / oversold bands
    fig.add_hline(y=80, line_dash="dot",
                  line=dict(color="rgba(239,83,80,0.4)", width=1))
    fig.add_hline(y=20, line_dash="dot",
                  line=dict(color="rgba(38,166,154,0.4)", width=1))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=55, r=80, t=28, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
        height=180,
        title=dict(text="<b>Stochastic (14, 3)</b>",
                   font=dict(size=12, color="#90a4ae"), x=0.01),
        yaxis=dict(range=[0, 100], gridcolor=_GRID, tickfont=dict(size=9)),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(size=9)),
    )
    return fig
