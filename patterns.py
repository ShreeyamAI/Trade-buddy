"""
Chart Pattern Detection Engine for Trade Buddy.
Detects 35+ bullish and bearish patterns from OHLCV candlestick data.

Patterns are grouped into:
  - Single-candle: Doji, Hammer, Shooting Star, Marubozu, Spinning Top ...
  - Two-candle: Engulfing, Harami, Piercing, Dark Cloud Cover, Tweezer ...
  - Three-candle: Morning Star, Evening Star, Three Soldiers / Crows ...
  - Price Structure: Double Top/Bottom, Head & Shoulders, Triangles, Wedges,
                     Flags, Breakouts, Golden/Death Cross, Support & Resistance

Each pattern returns a directional bias and a numeric score contribution.
No Machine Learning — purely rule-based mathematical detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# ──────────────────────────────────────────────
# Candle Geometry Helpers
# ──────────────────────────────────────────────

def _body(r) -> float:
    return abs(float(r['close']) - float(r['open']))

def _rng(r) -> float:
    return float(r['high']) - float(r['low'])

def _upper_shadow(r) -> float:
    return float(r['high']) - max(float(r['open']), float(r['close']))

def _lower_shadow(r) -> float:
    return min(float(r['open']), float(r['close'])) - float(r['low'])

def _is_bull(r) -> bool:
    return float(r['close']) >= float(r['open'])

def _is_bear(r) -> bool:
    return float(r['close']) < float(r['open'])


# ──────────────────────────────────────────────
# SINGLE-CANDLE PATTERNS
# ──────────────────────────────────────────────

def is_doji(r, threshold: float = 0.06) -> bool:
    """Doji: open ≈ close. Indecision candle."""
    rng = _rng(r)
    return rng > 0 and (_body(r) / rng) <= threshold

def is_spinning_top(r) -> bool:
    """Spinning Top: small body with comparable upper & lower shadows."""
    rng = _rng(r)
    if rng == 0:
        return False
    return (_body(r) / rng) <= 0.3 and _upper_shadow(r) >= 0.2 * rng and _lower_shadow(r) >= 0.2 * rng

def is_hammer(r) -> bool:
    """
    Hammer (bullish reversal): small body near top of range,
    lower shadow ≥ 2x body, tiny upper shadow.
    """
    b, lo, up, rng = _body(r), _lower_shadow(r), _upper_shadow(r), _rng(r)
    if rng == 0 or b == 0:
        return False
    return lo >= 2 * b and up <= 0.15 * rng and b <= 0.35 * rng

def is_hanging_man(r) -> bool:
    """Hanging Man (bearish reversal at top): same shape as Hammer."""
    return is_hammer(r)

def is_inverted_hammer(r) -> bool:
    """
    Inverted Hammer (bullish potential): small body near bottom,
    upper shadow ≥ 2x body, tiny lower shadow.
    """
    b, lo, up, rng = _body(r), _lower_shadow(r), _upper_shadow(r), _rng(r)
    if rng == 0 or b == 0:
        return False
    return up >= 2 * b and lo <= 0.15 * rng and b <= 0.35 * rng

def is_shooting_star(r) -> bool:
    """Shooting Star (bearish reversal at top): same shape as Inverted Hammer."""
    return is_inverted_hammer(r)

def is_marubozu_bull(r) -> bool:
    """Bullish Marubozu: large bullish body, almost no shadows."""
    rng = _rng(r)
    return _is_bull(r) and rng > 0 and (_body(r) / rng) >= 0.90

def is_marubozu_bear(r) -> bool:
    """Bearish Marubozu: large bearish body, almost no shadows."""
    rng = _rng(r)
    return _is_bear(r) and rng > 0 and (_body(r) / rng) >= 0.90

def is_long_legged_doji(r) -> bool:
    """Long-Legged Doji: open ≈ close, very long shadows on both sides."""
    rng = _rng(r)
    if rng == 0:
        return False
    return is_doji(r) and _upper_shadow(r) >= 0.3 * rng and _lower_shadow(r) >= 0.3 * rng

def is_dragonfly_doji(r) -> bool:
    """Dragonfly Doji: open ≈ close at top, very long lower shadow (bullish)."""
    rng = _rng(r)
    if rng == 0:
        return False
    return is_doji(r) and _lower_shadow(r) >= 0.6 * rng and _upper_shadow(r) <= 0.1 * rng

def is_gravestone_doji(r) -> bool:
    """Gravestone Doji: open ≈ close at bottom, very long upper shadow (bearish)."""
    rng = _rng(r)
    if rng == 0:
        return False
    return is_doji(r) and _upper_shadow(r) >= 0.6 * rng and _lower_shadow(r) <= 0.1 * rng


# ──────────────────────────────────────────────
# TWO-CANDLE PATTERNS
# ──────────────────────────────────────────────

def is_bullish_engulfing(curr, prev) -> bool:
    """Bullish Engulfing: bearish candle, then a larger bullish candle that covers it."""
    return (
        _is_bear(prev) and _is_bull(curr) and
        float(curr['open']) <= float(prev['close']) and
        float(curr['close']) >= float(prev['open'])
    )

def is_bearish_engulfing(curr, prev) -> bool:
    """Bearish Engulfing: bullish candle, then a larger bearish candle that covers it."""
    return (
        _is_bull(prev) and _is_bear(curr) and
        float(curr['open']) >= float(prev['close']) and
        float(curr['close']) <= float(prev['open'])
    )

def is_bullish_harami(curr, prev) -> bool:
    """Bullish Harami: large bearish, then small bullish inside the prior body."""
    if not _is_bear(prev) or not _is_bull(curr):
        return False
    p_top, p_bot = float(prev['open']), float(prev['close'])
    c_top, c_bot = float(curr['close']), float(curr['open'])
    return c_top < p_top and c_bot > p_bot

def is_bearish_harami(curr, prev) -> bool:
    """Bearish Harami: large bullish, then small bearish inside the prior body."""
    if not _is_bull(prev) or not _is_bear(curr):
        return False
    p_top, p_bot = float(prev['close']), float(prev['open'])
    c_top, c_bot = float(curr['open']), float(curr['close'])
    return c_top < p_top and c_bot > p_bot

def is_piercing_line(curr, prev) -> bool:
    """Piercing Line: bearish candle followed by bullish closing above prior midpoint."""
    if not _is_bear(prev) or not _is_bull(curr):
        return False
    midpoint = (float(prev['open']) + float(prev['close'])) / 2
    return float(curr['open']) < float(prev['close']) and float(curr['close']) > midpoint

def is_dark_cloud_cover(curr, prev) -> bool:
    """Dark Cloud Cover: bullish candle followed by bearish closing below prior midpoint."""
    if not _is_bull(prev) or not _is_bear(curr):
        return False
    midpoint = (float(prev['open']) + float(prev['close'])) / 2
    return float(curr['open']) > float(prev['close']) and float(curr['close']) < midpoint

def is_tweezer_bottom(curr, prev) -> bool:
    """Tweezer Bottom: two candles with matching lows — strong double support test."""
    tol = 0.002 * float(curr['low'])
    return (
        _is_bear(prev) and _is_bull(curr) and
        abs(float(curr['low']) - float(prev['low'])) <= tol
    )

def is_tweezer_top(curr, prev) -> bool:
    """Tweezer Top: two candles with matching highs — strong double resistance test."""
    tol = 0.002 * float(curr['high'])
    return (
        _is_bull(prev) and _is_bear(curr) and
        abs(float(curr['high']) - float(prev['high'])) <= tol
    )

def is_on_neck(curr, prev) -> bool:
    """On Neck: bearish candle, then small bullish closing near (but not above) prior close."""
    if not _is_bear(prev) or not _is_bull(curr):
        return False
    tol = 0.001 * float(prev['close'])
    return abs(float(curr['close']) - float(prev['close'])) <= tol

def is_mat_hold_bull(curr, prev) -> bool:
    """Bullish Mat Hold hint: gap-up bullish candle after bearish inside prior."""
    return _is_bull(prev) and _is_bull(curr) and float(curr['open']) > float(prev['close'])


# ──────────────────────────────────────────────
# THREE-CANDLE PATTERNS
# ──────────────────────────────────────────────

def is_morning_star(c3, c2, c1) -> bool:
    """Morning Star: large bearish | small body | large bullish (bullish reversal)."""
    if not _is_bear(c1) or not _is_bull(c3):
        return False
    b1, b2, b3 = _body(c1), _body(c2), _body(c3)
    mid_c1 = (float(c1['open']) + float(c1['close'])) / 2
    return b1 > b2 * 2 and b3 > b2 * 2 and float(c3['close']) > mid_c1

def is_evening_star(c3, c2, c1) -> bool:
    """Evening Star: large bullish | small body | large bearish (bearish reversal)."""
    if not _is_bull(c1) or not _is_bear(c3):
        return False
    b1, b2, b3 = _body(c1), _body(c2), _body(c3)
    mid_c1 = (float(c1['open']) + float(c1['close'])) / 2
    return b1 > b2 * 2 and b3 > b2 * 2 and float(c3['close']) < mid_c1

def is_morning_doji_star(c3, c2, c1) -> bool:
    """Morning Doji Star: like Morning Star but c2 is a Doji (stronger signal)."""
    return is_morning_star(c3, c2, c1) and is_doji(c2)

def is_evening_doji_star(c3, c2, c1) -> bool:
    """Evening Doji Star: like Evening Star but c2 is a Doji (stronger signal)."""
    return is_evening_star(c3, c2, c1) and is_doji(c2)

def is_three_white_soldiers(c3, c2, c1) -> bool:
    """Three White Soldiers: three consecutive rising bullish candles with higher closes."""
    return (
        _is_bull(c1) and _is_bull(c2) and _is_bull(c3) and
        float(c2['close']) > float(c1['close']) and float(c3['close']) > float(c2['close']) and
        float(c2['open']) > float(c1['open']) and float(c3['open']) > float(c2['open'])
    )

def is_three_black_crows(c3, c2, c1) -> bool:
    """Three Black Crows: three consecutive falling bearish candles with lower closes."""
    return (
        _is_bear(c1) and _is_bear(c2) and _is_bear(c3) and
        float(c2['close']) < float(c1['close']) and float(c3['close']) < float(c2['close']) and
        float(c2['open']) < float(c1['open']) and float(c3['open']) < float(c2['open'])
    )

def is_three_inside_up(c3, c2, c1) -> bool:
    """Three Inside Up: Bearish Harami + confirming bullish candle (bullish reversal)."""
    return is_bullish_harami(c2, c1) and _is_bull(c3) and float(c3['close']) > float(c2['close'])

def is_three_inside_down(c3, c2, c1) -> bool:
    """Three Inside Down: Bullish Harami + confirming bearish candle (bearish reversal)."""
    return is_bearish_harami(c2, c1) and _is_bear(c3) and float(c3['close']) < float(c2['close'])

def is_abandoned_baby_bull(c3, c2, c1) -> bool:
    """Abandoned Baby (Bullish): Bearish | Gap-down Doji | Gap-up Bullish."""
    return (
        _is_bear(c1) and is_doji(c2) and _is_bull(c3) and
        float(c2['high']) < float(c1['low']) and
        float(c2['low']) < float(c3['open'])
    )

def is_abandoned_baby_bear(c3, c2, c1) -> bool:
    """Abandoned Baby (Bearish): Bullish | Gap-up Doji | Gap-down Bearish."""
    return (
        _is_bull(c1) and is_doji(c2) and _is_bear(c3) and
        float(c2['low']) > float(c1['high']) and
        float(c2['high']) > float(c3['open'])
    )


# ──────────────────────────────────────────────
# PRICE STRUCTURE PATTERNS (multi-bar)
# ──────────────────────────────────────────────

def _pivot_highs(df: pd.DataFrame, window: int = 3) -> pd.Index:
    """Returns indices of local swing highs."""
    w = window * 2 + 1
    return df.index[df['high'].rolling(w, center=True, min_periods=w).max() == df['high']]

def _pivot_lows(df: pd.DataFrame, window: int = 3) -> pd.Index:
    """Returns indices of local swing lows."""
    w = window * 2 + 1
    return df.index[df['low'].rolling(w, center=True, min_periods=w).min() == df['low']]

def detect_double_top(df: pd.DataFrame, tol: float = 0.006) -> Dict[str, Any]:
    """Double Top (bearish): two nearly equal swing highs separated by a trough."""
    recent = df.tail(70).reset_index(drop=True)
    piv = _pivot_highs(recent, window=4)
    if len(piv) < 2:
        return {"detected": False}
    i1, i2 = piv[-2], piv[-1]
    h1, h2 = float(recent.loc[i1, 'high']), float(recent.loc[i2, 'high'])
    if abs(h1 - h2) / h1 <= tol:
        neck = float(recent.iloc[i1:i2+1]['low'].min())
        return {"detected": True, "name": "Double Top", "bias": "bearish", "score": -2,
                "description": f"Two peaks ~{h1:.0f}/{h2:.0f} | Neckline: {neck:.0f}"}
    return {"detected": False}

def detect_double_bottom(df: pd.DataFrame, tol: float = 0.006) -> Dict[str, Any]:
    """Double Bottom (bullish): two nearly equal swing lows separated by a peak."""
    recent = df.tail(70).reset_index(drop=True)
    piv = _pivot_lows(recent, window=4)
    if len(piv) < 2:
        return {"detected": False}
    i1, i2 = piv[-2], piv[-1]
    l1, l2 = float(recent.loc[i1, 'low']), float(recent.loc[i2, 'low'])
    if abs(l1 - l2) / l1 <= tol:
        neck = float(recent.iloc[i1:i2+1]['high'].max())
        return {"detected": True, "name": "Double Bottom", "bias": "bullish", "score": +2,
                "description": f"Two troughs ~{l1:.0f}/{l2:.0f} | Neckline: {neck:.0f}"}
    return {"detected": False}

def detect_head_and_shoulders(df: pd.DataFrame) -> Dict[str, Any]:
    """Head & Shoulders (bearish): three peaks — middle highest, shoulders roughly equal."""
    recent = df.tail(90).reset_index(drop=True)
    piv = _pivot_highs(recent, window=5)
    highs = [float(recent.loc[i, 'high']) for i in piv]
    if len(highs) < 3:
        return {"detected": False}
    p1, p2, p3 = highs[-3], highs[-2], highs[-1]
    if p2 > p1 and p2 > p3 and abs(p1 - p3) / p2 <= 0.04:
        return {"detected": True, "name": "Head & Shoulders", "bias": "bearish", "score": -2,
                "description": f"LS≈{p1:.0f} | Head≈{p2:.0f} | RS≈{p3:.0f}"}
    return {"detected": False}

def detect_inverse_head_and_shoulders(df: pd.DataFrame) -> Dict[str, Any]:
    """Inverse H&S (bullish): three troughs — middle lowest, shoulders roughly equal."""
    recent = df.tail(90).reset_index(drop=True)
    piv = _pivot_lows(recent, window=5)
    lows = [float(recent.loc[i, 'low']) for i in piv]
    if len(lows) < 3:
        return {"detected": False}
    p1, p2, p3 = lows[-3], lows[-2], lows[-1]
    if p2 < p1 and p2 < p3 and abs(p1 - p3) / abs(p2) <= 0.04:
        return {"detected": True, "name": "Inverse H&S", "bias": "bullish", "score": +2,
                "description": f"LS≈{p1:.0f} | Head≈{p2:.0f} | RS≈{p3:.0f}"}
    return {"detected": False}

def detect_triangle(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Triangles using linear regression slope of highs vs lows:
    - Symmetrical: highs falling + lows rising
    - Ascending: flat highs + rising lows (bullish)
    - Descending: falling highs + flat lows (bearish)
    """
    if len(df) < 30:
        return {"detected": False}
    recent = df.tail(40).reset_index(drop=True)
    x = np.arange(len(recent))
    hs = float(np.polyfit(x, recent['high'].values, 1)[0])
    ls = float(np.polyfit(x, recent['low'].values, 1)[0])
    tol = float(recent['close'].mean()) * 0.0003

    if abs(hs) <= tol and ls > tol:
        return {"detected": True, "name": "Ascending Triangle", "bias": "bullish", "score": +1,
                "description": "Flat resistance, rising support — bullish compression"}
    if hs < -tol and abs(ls) <= tol:
        return {"detected": True, "name": "Descending Triangle", "bias": "bearish", "score": -1,
                "description": "Falling highs, flat support — bearish compression"}
    if hs < -tol and ls > tol:
        return {"detected": True, "name": "Symmetrical Triangle", "bias": "neutral", "score": 0,
                "description": "Converging highs & lows — breakout imminent"}
    return {"detected": False}

def detect_wedge(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Wedges:
    - Rising Wedge (bearish): both highs & lows rise, but lows slope steeper
    - Falling Wedge (bullish): both highs & lows fall, but highs slope steeper
    """
    if len(df) < 30:
        return {"detected": False}
    recent = df.tail(40).reset_index(drop=True)
    x = np.arange(len(recent))
    hs = float(np.polyfit(x, recent['high'].values, 1)[0])
    ls = float(np.polyfit(x, recent['low'].values, 1)[0])

    if hs > 0 and ls > 0 and ls > hs * 1.3:
        return {"detected": True, "name": "Rising Wedge", "bias": "bearish", "score": -1,
                "description": "Both trendlines rising — support rising faster than resistance (bearish)"}
    if hs < 0 and ls < 0 and hs < ls * 1.3:
        return {"detected": True, "name": "Falling Wedge", "bias": "bullish", "score": +1,
                "description": "Both trendlines falling — resistance falling faster than support (bullish)"}
    return {"detected": False}

def detect_flag(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Flag Patterns: sharp impulse move followed by a tight consolidation.
    Bull Flag = bullish continuation; Bear Flag = bearish continuation.
    """
    if len(df) < 20:
        return {"detected": False}
    impulse = df.tail(30).head(15)
    consol = df.tail(15)
    move = float(impulse.iloc[-1]['close']) - float(impulse.iloc[0]['close'])
    imp_range = float(impulse['close'].max()) - float(impulse['close'].min())
    con_range = float(consol['close'].max()) - float(consol['close'].min())
    if imp_range > 0 and con_range < imp_range * 0.4:
        if move > 0:
            return {"detected": True, "name": "Bull Flag", "bias": "bullish", "score": +1,
                    "description": f"Sharp rally +{move:.0f} pts, now consolidating"}
        elif move < 0:
            return {"detected": True, "name": "Bear Flag", "bias": "bearish", "score": -1,
                    "description": f"Sharp drop {move:.0f} pts, now consolidating"}
    return {"detected": False}

def detect_cup_and_handle(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Cup & Handle (bullish): U-shaped price curve followed by a small downward drift.
    Approximated by checking that price forms a rounded bottom pattern.
    """
    if len(df) < 50:
        return {"detected": False}
    cup = df.tail(50).reset_index(drop=True)
    first_close = float(cup.iloc[0]['close'])
    last_close = float(cup.iloc[-1]['close'])
    bottom_min = float(cup.iloc[10:40]['low'].min())
    top_ref = min(first_close, last_close)
    depth = top_ref - bottom_min
    if depth > 0 and abs(first_close - last_close) / first_close <= 0.03 and depth / top_ref >= 0.01:
        return {"detected": True, "name": "Cup & Handle", "bias": "bullish", "score": +1,
                "description": f"Rounded bottom ({bottom_min:.0f}) with rim near {top_ref:.0f}"}
    return {"detected": False}

def detect_breakout(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """Breakout: price closes above/below the prior N-bar high/low."""
    if len(df) < lookback + 1:
        return {"detected": False}
    hist = df.iloc[-(lookback + 1):-1]
    curr = df.iloc[-1]
    rh = float(hist['high'].max())
    rl = float(hist['low'].min())
    if float(curr['close']) > rh:
        return {"detected": True, "name": "Bullish Breakout", "bias": "bullish", "score": +2,
                "description": f"Close {curr['close']:.0f} above {lookback}-bar high {rh:.0f}"}
    if float(curr['close']) < rl:
        return {"detected": True, "name": "Bearish Breakdown", "bias": "bearish", "score": -2,
                "description": f"Close {curr['close']:.0f} below {lookback}-bar low {rl:.0f}"}
    return {"detected": False}

def detect_ema_cross(df: pd.DataFrame) -> Dict[str, Any]:
    """Golden Cross / Death Cross on EMA 9 vs EMA 21."""
    if len(df) < 22 or 'ema_9' not in df.columns:
        return {"detected": False}
    c, p = df.iloc[-1], df.iloc[-2]
    if float(p['ema_9']) <= float(p['ema_21']) and float(c['ema_9']) > float(c['ema_21']):
        return {"detected": True, "name": "Golden Cross", "bias": "bullish", "score": +2,
                "description": f"EMA(9) crossed above EMA(21) at {c['close']:.2f}"}
    if float(p['ema_9']) >= float(p['ema_21']) and float(c['ema_9']) < float(c['ema_21']):
        return {"detected": True, "name": "Death Cross", "bias": "bearish", "score": -2,
                "description": f"EMA(9) crossed below EMA(21) at {c['close']:.2f}"}
    return {"detected": False}

def detect_support_resistance(df: pd.DataFrame) -> Dict[str, Any]:
    """Finds nearest support and resistance levels from recent swing pivots."""
    if len(df) < 20:
        return {"support": None, "resistance": None, "all_supports": [], "all_resistances": []}
    recent = df.tail(120) if len(df) >= 120 else df
    current = float(df.iloc[-1]['close'])
    tol = 0.004

    # Find swing highs and lows
    swing_highs, swing_lows = [], []
    for i in range(5, len(recent) - 5):
        row_h = float(recent.iloc[i]['high'])
        row_l = float(recent.iloc[i]['low'])
        if row_h == float(recent.iloc[i-5:i+6]['high'].max()):
            swing_highs.append(row_h)
        if row_l == float(recent.iloc[i-5:i+6]['low'].min()):
            swing_lows.append(row_l)

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(set(levels))
        out = [[levels[0]]]
        for v in levels[1:]:
            if v <= out[-1][-1] * (1 + tol):
                out[-1].append(v)
            else:
                out.append([v])
        return [round(float(np.mean(c)), 2) for c in out]

    res_levels = sorted([l for l in cluster(swing_highs) if l > current * (1 + tol * 0.5)])
    sup_levels = sorted([l for l in cluster(swing_lows) if l < current * (1 - tol * 0.5)], reverse=True)

    return {
        "support": sup_levels[0] if sup_levels else None,
        "resistance": res_levels[0] if res_levels else None,
        "all_supports": sup_levels[:5],
        "all_resistances": res_levels[:5],
        "near_support": bool(sup_levels and abs(current - sup_levels[0]) / current <= tol),
        "near_resistance": bool(res_levels and abs(current - res_levels[0]) / current <= tol),
    }


# ──────────────────────────────────────────────
# MASTER RUNNER
# ──────────────────────────────────────────────

def detect_all_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs all 35+ pattern detectors on the provided OHLCV DataFrame.
    Returns a unified dict with all detected patterns, net directional score,
    and support/resistance levels.
    """
    if df.empty or len(df) < 5:
        return {"patterns": [], "net_score": 0, "primary_bias": "neutral",
                "support_resistance": {}}

    detected: List[Dict] = []
    net_score = 0

    c0 = df.iloc[-1]   # current candle
    c1 = df.iloc[-2] if len(df) >= 2 else c0
    c2 = df.iloc[-3] if len(df) >= 3 else c1

    def add(name, bias, score, desc, typ="candlestick"):
        nonlocal net_score
        net_score += score
        detected.append({"name": name, "bias": bias, "score": score,
                         "description": desc, "type": typ})

    # ── Single Candle ──────────────────────────
    if is_dragonfly_doji(c0):
        add("Dragonfly Doji", "bullish", +1, "Open≈Close at top, long lower tail — buyers absorbed sellers")
    elif is_gravestone_doji(c0):
        add("Gravestone Doji", "bearish", -1, "Open≈Close at bottom, long upper tail — sellers absorbed buyers")
    elif is_long_legged_doji(c0):
        add("Long-Legged Doji", "neutral", 0, "Extreme indecision — long shadows on both sides")
    elif is_doji(c0):
        add("Doji", "neutral", 0, "Open ≈ Close — market indecision")
    if is_marubozu_bull(c0):
        add("Bullish Marubozu", "bullish", +1, "Full bullish body, no shadows — strong buyer dominance")
    elif is_marubozu_bear(c0):
        add("Bearish Marubozu", "bearish", -1, "Full bearish body, no shadows — strong seller dominance")
    elif is_spinning_top(c0):
        add("Spinning Top", "neutral", 0, "Small body, equal shadows — neither side in control")

    # Hammer vs Hanging Man: context-sensitive
    if is_hammer(c0) and _is_bear(c1):
        add("Hammer", "bullish", +1, "Long lower wick in downtrend — buyers rejecting lower prices")
    elif is_hanging_man(c0) and _is_bull(c1):
        add("Hanging Man", "bearish", -1, "Long lower wick after uptrend — weakness warning")

    # Shooting Star vs Inverted Hammer: context-sensitive
    if is_shooting_star(c0) and _is_bull(c1):
        add("Shooting Star", "bearish", -1, "Long upper wick after rally — sellers overwhelming buyers")
    elif is_inverted_hammer(c0) and _is_bear(c1):
        add("Inverted Hammer", "bullish", +1, "Long upper wick in downtrend — bulls testing recovery")

    # ── Two Candle ────────────────────────────
    if is_bullish_engulfing(c0, c1):
        add("Bullish Engulfing", "bullish", +2, "Current bullish candle fully covers prior bearish — buyers in control")
    if is_bearish_engulfing(c0, c1):
        add("Bearish Engulfing", "bearish", -2, "Current bearish candle fully covers prior bullish — sellers in control")
    if is_bullish_harami(c0, c1):
        add("Bullish Harami", "bullish", +1, "Small bullish body inside prior bearish — selling pressure waning")
    if is_bearish_harami(c0, c1):
        add("Bearish Harami", "bearish", -1, "Small bearish body inside prior bullish — buying pressure waning")
    if is_piercing_line(c0, c1):
        add("Piercing Line", "bullish", +1, "Bullish candle pierces above prior close midpoint")
    if is_dark_cloud_cover(c0, c1):
        add("Dark Cloud Cover", "bearish", -1, "Bearish candle closes into prior bullish midpoint")
    if is_tweezer_bottom(c0, c1):
        add("Tweezer Bottom", "bullish", +1, "Equal lows — strong double rejection of lower prices")
    if is_tweezer_top(c0, c1):
        add("Tweezer Top", "bearish", -1, "Equal highs — strong double rejection of higher prices")
    if is_mat_hold_bull(c0, c1):
        add("Gap-Up Continuation", "bullish", +1, "Gap-up open confirms upward momentum")

    # ── Three Candle ──────────────────────────
    if len(df) >= 3:
        if is_morning_doji_star(c0, c1, c2):
            add("Morning Doji Star", "bullish", +2, "Strong 3-bar bullish reversal with Doji indecision at bottom")
        elif is_morning_star(c0, c1, c2):
            add("Morning Star", "bullish", +2, "Classic 3-bar bullish reversal — downtrend exhaustion confirmed")

        if is_evening_doji_star(c0, c1, c2):
            add("Evening Doji Star", "bearish", -2, "Strong 3-bar bearish reversal with Doji indecision at top")
        elif is_evening_star(c0, c1, c2):
            add("Evening Star", "bearish", -2, "Classic 3-bar bearish reversal — uptrend exhaustion confirmed")

        if is_three_white_soldiers(c0, c1, c2):
            add("Three White Soldiers", "bullish", +2, "Three consecutive strong closes — sustained buying pressure")
        if is_three_black_crows(c0, c1, c2):
            add("Three Black Crows", "bearish", -2, "Three consecutive strong closes lower — sustained selling pressure")
        if is_three_inside_up(c0, c1, c2):
            add("Three Inside Up", "bullish", +2, "Harami confirmed by bullish third candle — buyers taking over")
        if is_three_inside_down(c0, c1, c2):
            add("Three Inside Down", "bearish", -2, "Harami confirmed by bearish third candle — sellers taking over")
        if is_abandoned_baby_bull(c0, c1, c2):
            add("Abandoned Baby (Bull)", "bullish", +2, "Gap isolation Doji — extremely rare strong reversal signal")
        if is_abandoned_baby_bear(c0, c1, c2):
            add("Abandoned Baby (Bear)", "bearish", -2, "Gap isolation Doji — extremely rare strong reversal signal")

    # ── Price Structure Patterns ──────────────
    for fn in [detect_double_top, detect_double_bottom, detect_head_and_shoulders,
               detect_inverse_head_and_shoulders, detect_triangle, detect_wedge,
               detect_flag, detect_cup_and_handle, detect_breakout, detect_ema_cross]:
        result = fn(df)
        if result.get("detected"):
            net_score += result.get("score", 0)
            detected.append({**result, "type": "price_structure"})

    # Support & Resistance
    sr = detect_support_resistance(df)

    # Final bias
    bias = "neutral"
    if net_score >= 2:
        bias = "bullish"
    elif net_score <= -2:
        bias = "bearish"

    return {
        "patterns": detected,
        "net_score": net_score,
        "primary_bias": bias,
        "support_resistance": sr
    }
