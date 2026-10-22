"""
Trade Buddy — Streamlit Dashboard v3
Clean UI · Live Prediction · Bot Confidence Display
Analysis-only. No broker. No execution.
"""

import os, time
import streamlit as st
import pandas as pd

from data_loader    import validate_and_load_csv, REQUIRED_COLUMNS
from analysis       import compute_indicators, evaluate_scenario
from patterns       import detect_all_patterns
from signals        import (
    compute_extended_indicators, mark_historical_signals,
    generate_signal, calculate_confidence, make_prediction,
)
from charts         import create_price_chart, create_macd_chart, create_stoch_chart
from live_data      import market_status, fetch_live_nifty
from news_sentiment import fetch_news
import database as db

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Buddy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* Slim sidebar */
[data-testid="stSidebar"] { min-width: 240px; max-width: 260px; }

/* Hero card */
.hero {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border-radius: 14px; padding: 22px 28px;
    border: 1px solid #2a3a55; margin-bottom: 18px;
}
.hero-price   { font-size: 2.6rem; font-weight: 700; line-height: 1.1; }
.hero-dir     { font-size: 1.15rem; font-weight: 600; margin-top: 4px; }
.hero-small   { font-size: .82rem; color: #90a4ae; margin-top: 6px; }

/* Confidence ring (HTML circle) */
.conf-ring {
    width: 110px; height: 110px; border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-weight: 700; border: 5px solid;
}
.conf-pct  { font-size: 1.9rem; line-height: 1; }
.conf-lbl  { font-size: .68rem; color: #90a4ae; text-transform: uppercase; }

/* Factor chip */
.fchip {
    border-radius: 7px; padding: 7px 11px;
    margin-bottom: 6px; font-size: .82rem; line-height: 1.3;
    border-left: 3px solid;
}

/* Pattern badge */
.pbull { background:#0d3826; border:1px solid #26a69a; border-radius:6px;
         padding:7px 11px; margin:3px 0; font-size:.82rem; }
.pbear { background:#3b0f12; border:1px solid #ef5350; border-radius:6px;
         padding:7px 11px; margin:3px 0; font-size:.82rem; }
.pneu  { background:#1a1f2e; border:1px solid #5c6bc0; border-radius:6px;
         padding:7px 11px; margin:3px 0; font-size:.82rem; }

/* News card */
.news-card { background:#131929; border-radius:8px; padding:10px 14px;
             border:1px solid #1e2c44; margin-bottom:8px; }
.news-bull  { border-left:3px solid #26a69a; }
.news-bear  { border-left:3px solid #ef5350; }
.news-neu   { border-left:3px solid #5c6bc0; }

/* Disclaimer */
.disc { background:#151e2e; border-left:4px solid #ff9800;
        padding:10px 14px; border-radius:4px;
        font-size:.8rem; color:#b0bec5; margin-bottom:14px; }

/* Remove Streamlit default padding on tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
""", unsafe_allow_html=True)

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_nifty50_1min.csv")


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
def build_sidebar():
    mkt = market_status()
    badge = "🟢" if mkt["is_open"] else "🔴"
    st.sidebar.markdown(f"### 📈 Trade Buddy")
    st.sidebar.markdown(f"{badge} **{mkt['message']}**  \n`{mkt['time_str']}`")
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Data source",
        ["📈 Live (yfinance)", "📂 Sample CSV", "📤 Upload CSV"],
        label_visibility="collapsed",
    )
    auto_ref = False
    ref_sec  = 30
    if mode == "📈 Live (yfinance)":
        if st.sidebar.button("🔄 Fetch Latest Candle Now", use_container_width=True):
            st.session_state["last_refresh"] = 0
            st.rerun()
        auto_ref = st.sidebar.toggle("Auto-refresh", value=True)
        ref_sec  = st.sidebar.select_slider("Refresh interval", [10, 15, 30, 60, 120, 300], value=30,
                                            format_func=lambda s: f"{s}s" if s < 60 else f"{s//60}m")

    st.sidebar.divider()
    horizon = st.sidebar.select_slider("Horizon (min)", [5, 10, 15], value=10)

    candle_mode = st.sidebar.radio("Evaluate", ["Latest candle", "Pick candle"],
                                   horizontal=True)

    st.sidebar.divider()
    st.sidebar.markdown("**Chart overlays**")
    show_bb      = st.sidebar.toggle("Bollinger Bands",    value=False)
    show_vwap    = st.sidebar.toggle("VWAP",               value=False)
    show_sr      = st.sidebar.toggle("Support/Resistance", value=True)
    show_signals = st.sidebar.toggle("Signal markers",     value=True)
    show_macd    = st.sidebar.toggle("MACD chart",         value=False)
    show_stoch   = st.sidebar.toggle("Stochastic chart",   value=False)
    chart_bars   = st.sidebar.slider("Bars to show", 30, 375, 120, 10)

    st.sidebar.divider()
    fetch_news_on = st.sidebar.toggle("📰 Live news", value=True,
                                      help="Reads free public RSS feeds")

    return dict(
        mode=mode, auto_ref=auto_ref, ref_sec=ref_sec,
        horizon=horizon, candle_mode=candle_mode,
        show_bb=show_bb, show_vwap=show_vwap, show_sr=show_sr,
        show_signals=show_signals, show_macd=show_macd, show_stoch=show_stoch,
        chart_bars=chart_bars, fetch_news=fetch_news_on,
    )


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data(opts):
    mode = opts["mode"]
    if mode == "📈 Live (yfinance)":
        with st.spinner("Fetching live NIFTY 50 data…"):
            df, status = fetch_live_nifty("1m", "1d")
        if df is None:
            return None, "Live", [status]
        st.caption(f"ℹ️ {status}")
        return df, "Yahoo Finance (^NSEI)", []

    if mode == "📂 Sample CSV":
        if not os.path.exists(SAMPLE_PATH):
            return None, "Sample", [f"File missing: {SAMPLE_PATH}"]
        df, errs = validate_and_load_csv(SAMPLE_PATH)
        return df, "Sample CSV", errs

    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up:
        df, errs = validate_and_load_csv(up)
        return df, up.name, errs
    st.info("👈 Upload a CSV or switch source in sidebar.")
    return None, "", []


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE RING (HTML)
# ═══════════════════════════════════════════════════════════════
def conf_ring_html(pct: int, color: str) -> str:
    # Simple circular badge using CSS border-radius + solid border trick
    # Approximate ring fill via conic-gradient
    bg = f"conic-gradient({color} {pct * 3.6}deg, #1e2638 0deg)"
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px">
      <div style="width:108px;height:108px;border-radius:50%;
                  background:{bg};
                  display:flex;align-items:center;justify-content:center;">
        <div style="width:80px;height:80px;border-radius:50%;
                    background:#131929;
                    display:flex;flex-direction:column;
                    align-items:center;justify-content:center;">
          <span style="color:{color};font-size:1.8rem;font-weight:700;line-height:1">{pct}%</span>
          <span style="color:#90a4ae;font-size:.62rem;text-transform:uppercase;letter-spacing:1px">confidence</span>
        </div>
      </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════
#  HERO PREDICTION CARD
# ═══════════════════════════════════════════════════════════════
def render_hero(pred: dict):
    """Full-width prediction hero at the top of the page."""
    dc  = pred["dir_color"]
    pct = pred["confidence"]
    sig = pred["signal_label"]
    cur = pred["current"]
    tgt = pred["target"]
    chg = pred["change"]
    pchg = pred["change_pct"]
    ts  = pred["timestamp"]
    hor = pred["horizon"]
    bf  = pred["bull_factors"]
    brf = pred["bear_factors"]
    cd  = pred["conf_desc"]
    atr = pred["atr"]
    risk = pred["risk_ref"]

    # Confidence ring color
    ring_col = dc

    # Direction arrow
    arrow = "▲" if "BULLISH" in pred["direction"] else ("▼" if "BEARISH" in pred["direction"] else "—")
    chg_str  = f"{arrow} ₹{abs(chg):,.2f} ({'+' if chg >= 0 else ''}{pchg:.2f}%)"
    risk_str = f"₹{risk:,.2f}" if risk else "—"

    # Build 3 columns inside the hero
    col_left, col_mid, col_right = st.columns([3, 2, 2])

    with col_left:
        st.markdown(f"""
        <div class="hero">
          <div style="color:#90a4ae;font-size:.78rem;letter-spacing:1px;text-transform:uppercase">
            Live Prediction · {hor}-min horizon · <code>{ts[-8:]}</code>
          </div>
          <div class="hero-price" style="color:{dc}">{pred['direction']}</div>
          <div class="hero-dir" style="color:#e0e0e0">
            Target &nbsp;<span style="color:{dc}">₹{tgt:,.2f}</span>
            &nbsp;<span style="color:#90a4ae">{chg_str}</span>
          </div>
          <div class="hero-small">
            Current: ₹{cur:,.2f} &nbsp;|&nbsp;
            ATR(14): {atr} pts &nbsp;|&nbsp;
            Reference stop: {risk_str}
          </div>
          <div class="hero-small" style="margin-top:6px">
            <span style="background:#1e2638;border-radius:4px;padding:2px 8px">{sig}</span>
            &nbsp;&nbsp;{bf} bullish · {brf} bearish factors
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_mid:
        st.markdown(
            f'<div style="display:flex;justify-content:center;align-items:center;height:100%;padding:10px 0">'
            + conf_ring_html(pct, ring_col)
            + f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align:center;color:#90a4ae;font-size:.78rem;margin-top:4px">{cd}</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        # Confidence breakdown bar
        st.markdown("**Factor scorecard**")
        bar_pct = min(100, max(0, (pred.get("confidence", 50) + 100) / 2))  # normalise
        bull_pct = int(bf / max(1, bf + brf) * 100)
        bear_pct = 100 - bull_pct
        st.markdown(f"""
        <div style="margin-top:8px">
          <div style="font-size:.75rem;color:#90a4ae;margin-bottom:4px">
            Bullish vs Bearish factors
          </div>
          <div style="display:flex;border-radius:6px;overflow:hidden;height:20px">
            <div style="background:#26a69a;width:{bull_pct}%;
                 display:flex;align-items:center;justify-content:center;
                 font-size:.7rem;color:#fff;font-weight:600">{bf}</div>
            <div style="background:#ef5350;width:{bear_pct}%;
                 display:flex;align-items:center;justify-content:center;
                 font-size:.7rem;color:#fff;font-weight:600">{brf}</div>
          </div>
          <div style="display:flex;justify-content:space-between;
               font-size:.7rem;color:#90a4ae;margin-top:2px">
            <span>▲ Bullish</span><span>▼ Bearish</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Confidence**")
        # Progress bar
        bar_c = dc
        st.markdown(f"""
        <div style="background:#1e2638;border-radius:6px;height:18px;overflow:hidden;margin-top:4px">
          <div style="background:{bar_c};width:{pct}%;height:100%;
               display:flex;align-items:center;padding-left:8px;
               color:#fff;font-size:.73rem;font-weight:700">
            {pct}%
          </div>
        </div>
        <div style="font-size:.72rem;color:#90a4ae;margin-top:3px">{cd}</div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  FACTOR CHIPS (side column)
# ═══════════════════════════════════════════════════════════════
def render_factors(signal_result: dict, pattern_result: dict):
    factors = signal_result.get("factors", [])
    st.markdown("**Signal factors**", help="Each input that contributed to the composite signal")
    for f in factors:
        s = f.get("score", 0)
        bias = f.get("bias", "neutral")
        if bias == "bullish":
            border = "#26a69a"; arrow = "▲"
        elif bias == "bearish":
            border = "#ef5350"; arrow = "▼"
        else:
            border = "#5c6bc0"; arrow = "—"
        col = "#c8e6c9" if bias == "bullish" else ("#ffcdd2" if bias == "bearish" else "#c5cae9")
        st.markdown(f"""
        <div class="fchip" style="background:#111827;border-left-color:{border}">
          <span style="color:{col};font-weight:600">{arrow} {f['label']}</span>
          &nbsp;<span style="color:#546e7a;font-size:.75rem">[{s:+d}]</span><br/>
          <span style="color:#78909c">{f['value'][:65]}</span>
        </div>
        """, unsafe_allow_html=True)

    # Pattern summary chips
    pats = pattern_result.get("patterns", [])
    if pats:
        st.markdown("**Patterns**")
        for p in pats[:6]:
            bias = p.get("bias", "neutral")
            cls  = "pbull" if bias == "bullish" else ("pbear" if bias == "bearish" else "pneu")
            ic   = "▲" if bias == "bullish" else ("▼" if bias == "bearish" else "—")
            col  = "#69f0ae" if bias == "bullish" else ("#ff8a80" if bias == "bearish" else "#90a4ae")
            st.markdown(f"""
            <div class="{cls}">
              <span style="color:{col}">{ic} <strong>{p['name']}</strong></span>
              <span style="color:#78909c;font-size:.75rem"> [{p.get('score',0):+d}]</span><br/>
              <span style="color:#607d8b;font-size:.75rem">{p.get('description','')[:60]}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    opts = build_sidebar()
    db.init_db()

    # ── Load data ──────────────────────────────────────────
    df_raw, src_name, errors = load_data(opts)
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        return
    if df_raw is None or df_raw.empty:
        return

    # ── Compute all indicators ─────────────────────────────
    df = compute_indicators(df_raw)
    df = compute_extended_indicators(df)
    total = len(df)

    # ── Candle index ───────────────────────────────────────
    if opts["candle_mode"] == "Latest candle":
        idx = total - 1
    else:
        idx = st.sidebar.slider("Candle index", 26, total - 1, total - 1, format="# %d")

    # ── Chart slice + signal markers ───────────────────────
    start   = max(0, idx - opts["chart_bars"] + 1)
    df_view = df.iloc[start: idx + 1].copy()
    df_view = mark_historical_signals(df_view)

    # ── Analysis ───────────────────────────────────────────
    pat_result  = detect_all_patterns(df.iloc[:idx + 1])
    sig_result  = generate_signal(df.iloc[:idx + 1], pat_result, {}, candle_idx=-1)

    # News (cache 5 min)
    news = {}
    if opts["fetch_news"]:
        cache_age = time.time() - st.session_state.get("news_ts", 0)
        if cache_age > 300:
            with st.spinner("Fetching news…"):
                news = fetch_news()
            st.session_state["news_cache"] = news
            st.session_state["news_ts"]    = time.time()
        else:
            news = st.session_state.get("news_cache", {})

    # Confidence + prediction
    conf = calculate_confidence(sig_result, df.iloc[:idx + 1], pat_result, news, candle_idx=-1)
    pred = make_prediction(df.iloc[:idx + 1], sig_result, pat_result, news,
                           confidence=conf, horizon_minutes=opts["horizon"], candle_idx=-1)

    sr       = pat_result.get("support_resistance", {})
    sup_lvls = sr.get("all_supports", [])    if opts["show_sr"] else []
    res_lvls = sr.get("all_resistances", []) if opts["show_sr"] else []
    target_ts = str(df.iloc[idx]["timestamp"])

    # Auto-evaluate pending scenarios
    updated = db.update_evaluations(df)
    if updated:
        st.toast(f"✅ {updated} scenario(s) evaluated", icon="✅")

    # Auto-refresh status label
    if opts["auto_ref"] and "Live" in opts["mode"]:
        age = time.time() - st.session_state.get("last_refresh", 0)
        target_interval = opts.get("ref_sec", 30)
        remaining = max(0, int(target_interval - age))
        st.sidebar.caption(f"🔄 Next live fetch in {remaining}s")

    # ────────────────────────────────────────────────────────
    # PAGE HEADER
    # ────────────────────────────────────────────────────────
    col_title, col_src = st.columns([5, 2])
    with col_title:
        st.markdown("## 📈 Trade Buddy")
        st.caption("Analysis-only · No broker · No order execution")
    with col_src:
        row = df.iloc[idx]
        delta = float(row["close"]) - float(df.iloc[0]["open"])
        sign  = "+" if delta >= 0 else ""
        st.metric("NIFTY 50", f"₹{row['close']:,.2f}",
                  delta=f"{sign}{delta:.1f} pts session",
                  label_visibility="visible")

    st.markdown("""
    <div class="disc">
    🛡️ <strong>Research-only:</strong> Trade Buddy never executes orders or connects to a broker.
    All signals and predictions are experimental study indicators, not financial advice.
    </div>
    """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────
    # HERO CARD
    # ────────────────────────────────────────────────────────
    render_hero(pred)

    # ────────────────────────────────────────────────────────
    # CHART  +  FACTOR COLUMN
    # ────────────────────────────────────────────────────────
    chart_col, factor_col = st.columns([3, 1], gap="medium")

    with chart_col:
        fig = create_price_chart(
            df_view,
            show_bb=opts["show_bb"],
            show_vwap=opts["show_vwap"],
            show_signals=opts["show_signals"],
            show_sr=opts["show_sr"],
            support_levels=sup_lvls,
            resistance_levels=res_lvls,
            highlight_ts=target_ts,
            title=f"NIFTY 50 · 1-min · {src_name}",
        )
        plot_config = {
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
            "displaylogo": False,
        }
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

        # Optional MACD
        if opts["show_macd"]:
            st.plotly_chart(create_macd_chart(df_view), use_container_width=True, config=plot_config)

        # Optional Stoch
        if opts["show_stoch"]:
            st.plotly_chart(create_stoch_chart(df_view), use_container_width=True, config=plot_config)

    with factor_col:
        render_factors(sig_result, pat_result)

    # ────────────────────────────────────────────────────────
    # TABS
    # ────────────────────────────────────────────────────────
    tab_pred, tab_news, tab_track, tab_data = st.tabs([
        "📋 Full Prediction Rationale",
        "📰 News Sentiment",
        "📓 Paper Tracker",
        "🔬 Data Health",
    ])

    # ══ FULL PREDICTION RATIONALE ════════════════════════════
    with tab_pred:
        st.subheader("Complete Prediction Rationale")
        st.caption(
            f"Direction: **{pred['direction']}** | "
            f"Bot Confidence: **{pred['confidence']}%** | "
            f"Horizon: **{pred['horizon']} min** | "
            f"Based on {pred['bull_factors']+pred['bear_factors']} signal inputs + "
            f"{len(pat_result.get('patterns',[]))} chart patterns + news"
        )

        dc   = pred["dir_color"]
        pct  = pred["confidence"]
        c_desc = pred["conf_desc"]

        # Price info
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Current Price",  f"₹{pred['current']:,.2f}")
        p2.metric("Predicted Target", f"₹{pred['target']:,.2f}",
                  delta=f"{pred['change']:+.2f} pts")
        p3.metric("Change %",        f"{pred['change_pct']:+.3f}%")
        p4.metric("ATR (14)",         f"{pred['atr']} pts")
        p5.metric("Bot Confidence",   f"{pct}%")

        st.markdown("---")
        st.markdown(f"**Confidence:** {pct}% — *{c_desc}*")
        # Full confidence bar
        st.progress(pct / 100)

        st.markdown("#### All contributing factors:")
        for line in pred["rationale"]:
            arrow = line[0] if line else "—"
            col = "#69f0ae" if arrow == "▲" else ("#ff8a80" if arrow == "▼" else "#90a4ae")
            st.markdown(
                f"<div style='padding:5px 0;color:{col};font-size:.88rem'>{line}</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Log Prediction to Paper Tracker")
        note = st.text_input("Note / tag:", "Prediction study", key="pred_note")
        if st.button("💾 Save to Tracker", type="primary"):
            row_id = db.log_scenario(
                candle_timestamp=target_ts,
                candle_close=float(df.iloc[idx]["close"]),
                scenario=pred["signal_label"],
                horizon_minutes=opts["horizon"],
                factors={"confidence": conf, "target": pred["target"],
                         "score": sig_result.get("score", 0)},
                notes=note,
            )
            db.update_evaluations(df)
            st.success(f"Saved as #{row_id} — target ₹{pred['target']:,.2f} in {opts['horizon']} min")
            st.rerun()

    # ══ NEWS SENTIMENT ════════════════════════════════════════
    with tab_news:
        st.subheader("Indian Markets News Sentiment")
        if not opts["fetch_news"]:
            st.info("Enable 📰 Live news in the sidebar.")
        elif not news:
            st.warning("No news data yet.")
        else:
            n1, n2, n3 = st.columns(3)
            n1.metric("Aggregate Score",   f"{news.get('aggregate_score', 0):+.2f}")
            n2.metric("Market Mood",        news.get("sentiment_label", "—"))
            n3.metric("Articles / NIFTY-relevant",
                       f"{news.get('total_articles',0)} / {news.get('relevant_count',0)}")

            st.caption(f"Fetched at {news.get('fetched_at','—')} — Sources: ET Markets, BS, Moneycontrol, LiveMint")

            errs = news.get("errors", [])
            if errs:
                with st.expander(f"⚠️ {len(errs)} source error(s)"):
                    for e in errs:
                        st.warning(e)

            show_all = st.toggle("Show all (not just NIFTY-relevant)", value=False)
            arts = news.get("articles", [])
            display = arts if show_all else ([a for a in arts if a["relevant"]] or arts)

            for art in display[:18]:
                sc = art["sentiment"]
                bdr = "news-bull" if sc > 0.1 else ("news-bear" if sc < -0.1 else "news-neu")
                col = "#69f0ae" if sc > 0.1 else ("#ff8a80" if sc < -0.1 else "#90a4ae")
                rel = "🎯 " if art["relevant"] else ""
                st.markdown(f"""
                <div class="news-card {bdr}">
                  <div style="font-size:.85rem;font-weight:600;color:#e0e0e0">
                    {rel}{art['title'][:110]}
                  </div>
                  <div style="font-size:.75rem;color:#546e7a;margin:3px 0">
                    {art['source']} &nbsp;|&nbsp;
                    <span style="color:{col}">{art['sentiment_label']} ({sc:+.2f})</span>
                  </div>
                  {f'<div style="font-size:.78rem;color:#78909c">{art["summary"][:200]}</div>' if art.get("summary") else ""}
                </div>
                """, unsafe_allow_html=True)

            if st.button("🔄 Refresh news"):
                with st.spinner("Fetching…"):
                    fresh = fetch_news()
                st.session_state["news_cache"] = fresh
                st.session_state["news_ts"]    = time.time()
                st.rerun()

    # ══ PAPER TRACKER ════════════════════════════════════════
    with tab_track:
        st.subheader("Paper Tracking Log (SQLite)")
        st.caption("Local DB only. No returns implied. Research tracking.")

        stats = db.get_summary_statistics()
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Logged",      stats["total_logged"])
        t2.metric("Evaluated",   stats["evaluated"])
        t3.metric("Pending",     stats["pending"])
        t4.metric("Aligned",     f"{stats['aligned_count']} ({stats['alignment_rate_pct']}%)")
        t5.metric("Avg move",    f"{stats['avg_move_points']:+.2f} pts")

        bc, rc = st.columns(2)
        with bc:
            if st.button("🔄 Re-evaluate pending"):
                n = db.update_evaluations(df)
                st.info(f"{n} scenario(s) evaluated")
                st.rerun()
        with rc:
            if st.button("🗑️ Clear all logs"):
                db.clear_database()
                st.warning("Cleared.")
                st.rerun()

        logs = db.get_all_scenarios()
        if not logs.empty:
            cols = ["id", "candle_timestamp", "candle_close", "scenario",
                    "horizon_minutes", "target_timestamp", "evaluation_status",
                    "observed_price", "price_change", "outcome", "notes"]
            st.dataframe(logs[[c for c in cols if c in logs.columns]],
                         use_container_width=True, height=320)
        else:
            st.info("No logs yet — save a prediction from the 'Full Prediction Rationale' tab.")

    # ══ DATA HEALTH ══════════════════════════════════════════
    with tab_data:
        st.subheader("Data Health")
        dh1, dh2, dh3 = st.columns(3)
        with dh1:
            st.success(f"✅ Columns: {', '.join(REQUIRED_COLUMNS)}")
            st.write(f"Total rows: **{len(df):,}**")
        with dh2:
            st.success("✅ Timestamps: ascending")
            st.write(f"From `{df['timestamp'].iloc[0]}`")
            st.write(f"To `{df['timestamp'].iloc[-1]}`")
        with dh3:
            nulls = int(df[REQUIRED_COLUMNS].isnull().sum().sum())
            dupes = int(df["timestamp"].duplicated().sum())
            st.success("✅ No nulls") if nulls == 0 else st.error(f"{nulls} nulls")
            st.success("✅ No duplicates") if dupes == 0 else st.error(f"{dupes} dupes")

        view_cols = ["timestamp", "open", "high", "low", "close", "volume",
                     "ema_9", "ema_21", "rsi", "macd", "bb_upper", "bb_lower",
                     "stoch_k", "atr_14", "volume_ratio"]
        st.dataframe(df[[c for c in view_cols if c in df.columns]].tail(30),
                     use_container_width=True)

    # ── Non-blocking Live Auto-Refresh at End of Execution ──
    if opts["auto_ref"] and "Live" in opts["mode"]:
        target_interval = opts.get("ref_sec", 30)
        age = time.time() - st.session_state.get("last_refresh", 0)
        if age >= target_interval:
            st.session_state["last_refresh"] = time.time()
            st.rerun()
        else:
            # Sleep remainder of time without blocking page display, then rerun
            sleep_duration = max(1.0, min(float(target_interval - age), 5.0))
            time.sleep(sleep_duration)
            st.rerun()


if __name__ == "__main__":
    main()
