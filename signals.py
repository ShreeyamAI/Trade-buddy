"""
Composite Buy/Sell Signal Engine for Trade Buddy.
Aggregates 9 independent technical and sentiment inputs into one transparent signal.

Inputs scored:
  1. EMA Trend Alignment (EMA 9 vs EMA 21)
  2. RSI Momentum (level + direction)
  3. MACD Signal Line Crossover + Histogram
  4. Bollinger Band Position
  5. Stochastic Oscillator (%K / %D)
  6. Volume Confirmation
  7. VWAP Intraday Bias
  8. Chart Patterns (from patterns.py)
  9. News Sentiment (from news_sentiment.py)

OUTPUT LABELS:
  Score ≥ +5  →  STRONG BUY SIGNAL
  Score +2..4 →  BUY SIGNAL
  Score -1..1 →  HOLD / NEUTRAL
  Score -4..-2 → SELL SIGNAL
  Score ≤ -5  →  STRONG SELL SIGNAL

⚠️ ALL SIGNALS ARE FOR RESEARCH / STUDY ONLY. Not financial advice. No execution capability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Score thresholds
STRONG_BUY  =  5
BUY         =  2
SELL        = -2
STRONG_SELL = -5


# ──────────────────────────────────────────────
# Extended Indicator Calculator
# ──────────────────────────────────────────────

def compute_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds MACD, Bollinger Bands, Stochastic %K/%D, and VWAP to the DataFrame.
    Safe to call if analysis.py indicators (EMA, RSI, ATR) are already present.
    """
    df = df.copy()

    # MACD: EMA(12) - EMA(26), Signal EMA(9) of MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']   = df['ema_12'] - df['ema_26']
    df['macd_signal']    = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bollinger Bands: SMA(20) ± 2σ
    sma20 = df['close'].rolling(20, min_periods=5).mean()
    std20 = df['close'].rolling(20, min_periods=5).std()
    df['bb_mid']   = sma20
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    bb_width = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_pct']   = (df['close'] - df['bb_lower']) / bb_width  # 0 = lower band, 1 = upper
    df['bb_pct']   = df['bb_pct'].fillna(0.5)

    # Stochastic Oscillator: %K (fast) and %D (smoothed)
    lo14 = df['low'].rolling(14,  min_periods=5).min()
    hi14 = df['high'].rolling(14, min_periods=5).max()
    denom = (hi14 - lo14).replace(0, np.nan)
    df['stoch_k'] = ((df['close'] - lo14) / denom * 100).fillna(50.0)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(50.0)

    # Cumulative VWAP approximation (intraday from first bar)
    typical = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical * df['volume']).cumsum() / df['volume'].cumsum()

    return df


# ──────────────────────────────────────────────
# Historical Signal Marker Generator (vectorized)
# ──────────────────────────────────────────────

def mark_historical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'signal_type' column ('buy' | 'sell' | None) to every candle row using
    vectorized crossover detection. Much faster than a per-row loop.
    Used to draw buy/sell triangle markers on the chart.
    """
    df = df.copy()
    df['signal_type'] = None

    if len(df) < 27:
        return df

    prev = df.shift(1)

    # 1. EMA Golden/Death Cross
    if 'ema_9' in df.columns and 'ema_21' in df.columns:
        gold = (prev['ema_9'] <= prev['ema_21']) & (df['ema_9'] > df['ema_21'])
        dead = (prev['ema_9'] >= prev['ema_21']) & (df['ema_9'] < df['ema_21'])
        df.loc[gold, 'signal_type'] = 'buy'
        df.loc[dead, 'signal_type'] = 'sell'

    # 2. MACD Signal Line Crossover (only where no EMA cross already marked)
    if 'macd' in df.columns:
        macd_bull = (prev['macd'] <= prev['macd_signal']) & (df['macd'] > df['macd_signal'])
        macd_bear = (prev['macd'] >= prev['macd_signal']) & (df['macd'] < df['macd_signal'])
        df.loc[macd_bull & df['signal_type'].isna(), 'signal_type'] = 'buy'
        df.loc[macd_bear & df['signal_type'].isna(), 'signal_type'] = 'sell'

    # 3. RSI Oversold/Overbought cross-back
    if 'rsi' in df.columns:
        rsi_bull = (prev['rsi'] <= 30) & (df['rsi'] > 30)
        rsi_bear = (prev['rsi'] >= 70) & (df['rsi'] < 70)
        df.loc[rsi_bull & df['signal_type'].isna(), 'signal_type'] = 'buy'
        df.loc[rsi_bear & df['signal_type'].isna(), 'signal_type'] = 'sell'

    # 4. Stochastic crossover in extreme zones
    if 'stoch_k' in df.columns:
        s_bull = (prev['stoch_k'] <= prev['stoch_d']) & (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] < 25)
        s_bear = (prev['stoch_k'] >= prev['stoch_d']) & (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'] > 75)
        df.loc[s_bull & df['signal_type'].isna(), 'signal_type'] = 'buy'
        df.loc[s_bear & df['signal_type'].isna(), 'signal_type'] = 'sell'

    return df


# ──────────────────────────────────────────────
# Composite Signal Generator
# ──────────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
    pattern_result: Dict[str, Any],
    news_result: Dict[str, Any],
    candle_idx: int = -1,
) -> Dict[str, Any]:
    """
    Generates a full composite signal for a specific candle.
    Combines 9 independent inputs, returns a transparent scored breakdown.
    """
    if df.empty or len(df) < 10:
        return _empty_signal()

    idx = (len(df) + candle_idx) if candle_idx < 0 else candle_idx
    idx = max(0, min(idx, len(df) - 1))

    curr = df.iloc[idx]
    prev = df.iloc[max(0, idx - 1)]

    score = 0
    factors: List[Dict] = []

    def add(label, value, s, bias):
        nonlocal score
        score += s
        factors.append({"label": label, "value": value, "score": s, "bias": bias})

    # 1. EMA Trend Alignment
    if all(c in df.columns for c in ['ema_9', 'ema_21']):
        e9, e21, px = float(curr['ema_9']), float(curr['ema_21']), float(curr['close'])
        if px > e9 > e21:
            add("EMA Trend", f"Bullish Alignment Price({px:.0f}) > EMA9({e9:.0f}) > EMA21({e21:.0f})", +2, "bullish")
        elif px < e9 < e21:
            add("EMA Trend", f"Bearish Alignment Price({px:.0f}) < EMA9({e9:.0f}) < EMA21({e21:.0f})", -2, "bearish")
        elif px > e9:
            add("EMA Trend", f"Mild Bullish – Price above EMA9", +1, "bullish")
        elif px < e9:
            add("EMA Trend", f"Mild Bearish – Price below EMA9", -1, "bearish")

    # 2. RSI Level + Direction
    if 'rsi' in df.columns:
        rsi = float(curr['rsi'])
        p_rsi = float(prev['rsi'])
        if 52 <= rsi <= 65:
            add("RSI", f"Bullish momentum zone (RSI {rsi:.1f})", +2, "bullish")
        elif 35 <= rsi <= 48:
            add("RSI", f"Bearish momentum zone (RSI {rsi:.1f})", -2, "bearish")
        elif rsi > 72:
            add("RSI", f"Overbought — exhaustion risk (RSI {rsi:.1f})", -1, "bearish")
        elif rsi < 28:
            add("RSI", f"Oversold — bounce potential (RSI {rsi:.1f})", +1, "bullish")
        # RSI direction bonus
        if rsi > p_rsi and rsi < 70:
            add("RSI Direction", f"RSI rising {p_rsi:.0f}→{rsi:.0f}", +1, "bullish")
        elif rsi < p_rsi and rsi > 30:
            add("RSI Direction", f"RSI falling {p_rsi:.0f}→{rsi:.0f}", -1, "bearish")

    # 3. MACD Crossover + Histogram
    if 'macd' in df.columns:
        m, ms, mh = float(curr['macd']), float(curr['macd_signal']), float(curr['macd_histogram'])
        pm, pms = float(prev['macd']), float(prev['macd_signal'])
        pmh = float(prev['macd_histogram'])
        # Crossover event
        if pm <= pms and m > ms:
            add("MACD", "Bullish crossover — MACD crossed above signal line", +2, "bullish")
        elif pm >= pms and m < ms:
            add("MACD", "Bearish crossover — MACD crossed below signal line", -2, "bearish")
        # Histogram momentum
        if mh > 0 and mh > pmh:
            add("MACD Histogram", f"Expanding positive histogram ({mh:.2f})", +1, "bullish")
        elif mh < 0 and mh < pmh:
            add("MACD Histogram", f"Expanding negative histogram ({mh:.2f})", -1, "bearish")

    # 4. Bollinger Band Position
    if 'bb_pct' in df.columns:
        bp = float(curr['bb_pct'])
        bl = float(curr.get('bb_lower', curr['close']))
        bu = float(curr.get('bb_upper', curr['close']))
        if bp <= 0.08:
            add("Bollinger Bands", f"Price hugging lower band ({bl:.0f}) — potential mean-reversion zone", +2, "bullish")
        elif bp >= 0.92:
            add("Bollinger Bands", f"Price at upper band ({bu:.0f}) — potential resistance / overbought", -1, "bearish")
        elif bp >= 0.55:
            add("Bollinger Bands", f"Price in upper half of channel (BB%: {bp:.2f})", +1, "bullish")
        elif bp <= 0.45:
            add("Bollinger Bands", f"Price in lower half of channel (BB%: {bp:.2f})", -1, "bearish")

    # 5. Stochastic Oscillator
    if 'stoch_k' in df.columns:
        sk, sd = float(curr['stoch_k']), float(curr['stoch_d'])
        psk, psd = float(prev['stoch_k']), float(prev['stoch_d'])
        if psk <= psd and sk > sd and sk < 30:
            add("Stochastic", f"Bullish crossover in oversold zone (%K {sk:.0f})", +2, "bullish")
        elif psk >= psd and sk < sd and sk > 70:
            add("Stochastic", f"Bearish crossover in overbought zone (%K {sk:.0f})", -2, "bearish")
        elif sk < 20:
            add("Stochastic", f"Oversold (%K {sk:.0f})", +1, "bullish")
        elif sk > 80:
            add("Stochastic", f"Overbought (%K {sk:.0f})", -1, "bearish")

    # 6. Volume Confirmation
    if 'volume_ratio' in df.columns:
        vr = float(curr.get('volume_ratio', 1.0))
        bull_candle = float(curr['close']) >= float(curr['open'])
        if vr >= 1.3:
            if bull_candle:
                add("Volume", f"High volume ({vr:.2f}x) on bullish close — strong buying", +1, "bullish")
            else:
                add("Volume", f"High volume ({vr:.2f}x) on bearish close — strong selling", -1, "bearish")
        elif vr < 0.7:
            add("Volume", f"Thin volume move ({vr:.2f}x) — low conviction", 0, "neutral")

    # 7. VWAP
    if 'vwap' in df.columns:
        vwap = float(curr['vwap'])
        px = float(curr['close'])
        if px > vwap * 1.001:
            add("VWAP", f"Price above VWAP ({vwap:.0f}) — intraday bullish bias", +1, "bullish")
        elif px < vwap * 0.999:
            add("VWAP", f"Price below VWAP ({vwap:.0f}) — intraday bearish bias", -1, "bearish")

    # 8. Chart Patterns
    pat_score = int(pattern_result.get("net_score", 0))
    if pat_score != 0:
        clamped = max(-3, min(3, pat_score))
        names = [p['name'] for p in pattern_result.get("patterns", [])[:4]]
        bias = "bullish" if clamped > 0 else "bearish"
        add("Chart Patterns",
            f"{', '.join(names) if names else 'None'} (net: {pat_score:+d})",
            clamped, bias)

    # 9. News Sentiment
    if news_result and news_result.get("total_articles", 0) > 0:
        ns = float(news_result.get("aggregate_score", 0))
        label = news_result.get("sentiment_label", "")
        ns_score = +1 if ns > 0.12 else (-1 if ns < -0.12 else 0)
        add("News Sentiment",
            f"Aggregate {ns:+.2f} {label} ({news_result['total_articles']} articles)",
            ns_score,
            "bullish" if ns_score > 0 else ("bearish" if ns_score < 0 else "neutral"))

    # ── Final Label ──────────────────────────
    if score >= STRONG_BUY:
        label, color = "STRONG BUY 🚀", "#00e676"
        conf = "High"
    elif score >= BUY:
        label, color = "BUY 📈", "#69f0ae"
        conf = "Medium"
    elif score <= STRONG_SELL:
        label, color = "STRONG SELL 🔻", "#ff1744"
        conf = "High"
    elif score <= SELL:
        label, color = "SELL 📉", "#ef9a9a"
        conf = "Medium"
    else:
        label, color = "HOLD / NEUTRAL ⚖️", "#90a4ae"
        conf = "Low"

    bull_f = [f for f in factors if f['bias'] == 'bullish']
    bear_f = [f for f in factors if f['bias'] == 'bearish']

    return {
        "signal": label,
        "score": score,
        "color": color,
        "confidence": conf,
        "factors": factors,
        "bull_count": len(bull_f),
        "bear_count": len(bear_f),
        "max_score": 20,
        "disclaimer": (
            "⚠️ STUDY SIGNAL ONLY — Not financial advice. "
            "Trade Buddy never executes orders or connects to brokers. "
            "All analysis is experimental."
        ),
    }


def _empty_signal() -> Dict[str, Any]:
    return {
        "signal": "HOLD / NEUTRAL ⚖️", "score": 0, "color": "#90a4ae",
        "confidence": "Low", "factors": [], "bull_count": 0, "bear_count": 0,
        "max_score": 20,
        "disclaimer": "Insufficient data for signal generation.",
    }


# ──────────────────────────────────────────────
# CONFIDENCE CALCULATOR
# ──────────────────────────────────────────────

def calculate_confidence(
    signal_result: Dict[str, Any],
    df: pd.DataFrame,
    pattern_result: Dict[str, Any],
    news_result: Dict[str, Any],
    candle_idx: int = -1,
) -> int:
    """
    Calculates 0–95% confidence for the current signal.
    Based on: score magnitude, factor alignment, RSI health, volume, patterns, news alignment.
    Capped at 95% — no prediction is ever certain.
    """
    signal = signal_result.get("signal", "HOLD / NEUTRAL ⚖️")
    if "NEUTRAL" in signal or "HOLD" in signal:
        return 30

    is_bull   = "BUY" in signal
    score     = abs(signal_result.get("score", 0))
    max_score = signal_result.get("max_score", 20)

    # 1. Magnitude 0–50 pts
    magnitude = min(50, (score / max(1, max_score)) * 80)

    # 2. Factor alignment 0–25 pts
    b = signal_result.get("bull_count", 0)
    r = signal_result.get("bear_count", 0)
    total = b + r
    aligned   = b if is_bull else r
    align_pts = (aligned / max(1, total)) * 25 if total else 0

    # 3. Quality adjustments
    adj = 0
    idx = (len(df) + candle_idx) if candle_idx < 0 else candle_idx
    idx = max(0, min(idx, len(df) - 1))
    row = df.iloc[idx]

    if "rsi" in df.columns:
        rsi = float(row.get("rsi", 50))
        if is_bull:
            if rsi > 73:         adj -= 12
            elif 48 < rsi < 65:  adj +=  8
            elif rsi < 30:       adj +=  4
        else:
            if rsi < 27:         adj -= 12
            elif 35 < rsi < 52:  adj +=  8
            elif rsi > 70:       adj +=  4

    if "volume_ratio" in df.columns:
        vr = float(row.get("volume_ratio", 1.0))
        if vr > 1.6:    adj += 10
        elif vr > 1.2:  adj +=  5
        elif vr < 0.65: adj -=  8

    if "macd" in df.columns:
        macd = float(row.get("macd", 0))
        msig = float(row.get("macd_signal", 0))
        if (is_bull and macd > msig) or (not is_bull and macd < msig):
            adj += 6
        else:
            adj -= 4

    pat_net = abs(pattern_result.get("net_score", 0))
    adj += min(8, pat_net * 2)

    if news_result and news_result.get("total_articles", 0) > 0:
        ns = float(news_result.get("aggregate_score", 0))
        if (is_bull and ns > 0.1) or (not is_bull and ns < -0.1):
            adj += 5
        elif (is_bull and ns < -0.1) or (not is_bull and ns > 0.1):
            adj -= 5

    return int(min(95, max(20, magnitude + align_pts + adj)))


def confidence_description(pct: int) -> str:
    if pct >= 85: return "Very High — strong multi-factor agreement"
    if pct >= 70: return "High — most inputs aligned"
    if pct >= 55: return "Medium — partial agreement"
    if pct >= 40: return "Low — mixed signals"
    return "Very Low — conflicting signals"


# ──────────────────────────────────────────────
# LIVE PREDICTION MAKER
# ──────────────────────────────────────────────

def make_prediction(
    df: pd.DataFrame,
    signal_result: Dict[str, Any],
    pattern_result: Dict[str, Any],
    news_result: Dict[str, Any],
    confidence: int,
    horizon_minutes: int = 10,
    candle_idx: int = -1,
) -> Dict[str, Any]:
    """
    Synthesises all factors into one forward-looking prediction with ATR-based price target.
    ⚠️ Experimental study output only — not a guaranteed price forecast.
    """
    if df.empty or len(df) < 10:
        return _empty_prediction()

    idx     = (len(df) + candle_idx) if candle_idx < 0 else candle_idx
    idx     = max(0, min(idx, len(df) - 1))
    row     = df.iloc[idx]

    current = float(row["close"])
    atr     = float(row.get("atr_14", current * 0.001))
    signal  = signal_result.get("signal", "HOLD / NEUTRAL ⚖️")
    score   = signal_result.get("score", 0)
    max_s   = signal_result.get("max_score", 20)

    score_factor   = abs(score) / max(1, max_s)
    horizon_factor = horizon_minutes / 10.0
    conf_factor    = confidence / 100.0
    atr_mult       = 0.4 + score_factor * 1.2 * horizon_factor * conf_factor

    is_bull = "BUY" in signal
    is_bear = "SELL" in signal

    if is_bull:
        direction = "BULLISH 📈"; dir_color = "#69f0ae"; dir_short = "UP"
        target    = round(current + atr * atr_mult, 2)
        risk_ref  = round(current - atr * 0.6, 2)
    elif is_bear:
        direction = "BEARISH 📉"; dir_color = "#ff8a80"; dir_short = "DOWN"
        target    = round(current - atr * atr_mult, 2)
        risk_ref  = round(current + atr * 0.6, 2)
    else:
        direction = "SIDEWAYS ⚖️"; dir_color = "#90a4ae"; dir_short = "FLAT"
        target = current; risk_ref = None

    change     = round(target - current, 2)
    change_pct = round((change / current) * 100, 3) if current else 0

    # Full transparent rationale
    rationale: List[str] = []
    for f in signal_result.get("factors", []):
        if f.get("score", 0) != 0:
            arrow = "▲" if f["score"] > 0 else "▼"
            rationale.append(f"{arrow} {f['label']}: {f['value'][:70]}")

    pats = pattern_result.get("patterns", [])
    bull_pats = [p["name"] for p in pats if p["bias"] == "bullish"]
    bear_pats = [p["name"] for p in pats if p["bias"] == "bearish"]
    if bull_pats:
        rationale.append(f"▲ Bullish patterns: {', '.join(bull_pats[:4])}")
    if bear_pats:
        rationale.append(f"▼ Bearish patterns: {', '.join(bear_pats[:4])}")

    if news_result and news_result.get("total_articles", 0) > 0:
        ns  = news_result.get("market_sentiment", "neutral")
        agg = float(news_result.get("aggregate_score", 0))
        arr = "▲" if agg > 0.05 else ("▼" if agg < -0.05 else "—")
        rationale.append(f"{arr} News ({news_result.get('total_articles')} articles): {ns.title()} ({agg:+.2f})")

    return {
        "direction": direction, "dir_color": dir_color, "dir_short": dir_short,
        "current": current, "target": target, "change": change, "change_pct": change_pct,
        "risk_ref": risk_ref, "atr": round(atr, 2), "horizon": horizon_minutes,
        "confidence": confidence, "conf_desc": confidence_description(confidence),
        "bull_factors": signal_result.get("bull_count", 0),
        "bear_factors": signal_result.get("bear_count", 0),
        "rationale": rationale, "signal_label": signal,
        "timestamp": str(row["timestamp"]),
    }


def _empty_prediction() -> Dict[str, Any]:
    return {
        "direction": "SIDEWAYS ⚖️", "dir_color": "#90a4ae", "dir_short": "FLAT",
        "current": 0, "target": 0, "change": 0, "change_pct": 0,
        "risk_ref": None, "atr": 0, "horizon": 10,
        "confidence": 0, "conf_desc": "Insufficient data",
        "bull_factors": 0, "bear_factors": 0, "rationale": [], "signal_label": "",
        "timestamp": "",
    }
