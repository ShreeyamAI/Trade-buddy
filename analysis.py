"""
Technical Analysis and Scenario Assessment Engine for Trade Buddy.
Provides transparent, rule-based evaluations for short-term NIFTY 50 price action.
Computes price direction, momentum (RSI), volume relative to baseline, and short-term volatility (ATR).
Produces one of: 'upward scenario', 'downward scenario', or 'no clear setup'.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes required indicators on the candle DataFrame:
    - EMA(9) and EMA(21) for trend direction
    - RSI(14) for momentum
    - Volume 20-period Moving Average and Volume Ratio
    - True Range and ATR(14) for volatility
    """
    df = df.copy()

    # Exponential Moving Averages for trend
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

    # 14-period Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing for RSI
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Fill starting periods
    df['rsi'] = df['rsi'].fillna(50.0)

    # Volume comparison: 20-period rolling average
    df['vol_ma_20'] = df['volume'].rolling(window=20, min_periods=5).mean()
    df['volume_ratio'] = df['volume'] / df['vol_ma_20'].replace(0, np.nan)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

    # Volatility: True Range (TR) and Average True Range (ATR 14)
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - df['close'].shift(1)).abs()
    low_prev_close = (df['low'] - df['close'].shift(1)).abs()
    
    df['true_range'] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14, min_periods=5).mean()
    df['atr_14'] = df['atr_14'].fillna(df['high'] - df['low'])
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100.0

    return df


def evaluate_scenario(df: pd.DataFrame, candle_idx: int = -1) -> Dict[str, Any]:
    """
    Evaluates the market scenario for a specific candle index (defaults to the latest candle).
    
    Inputs evaluated:
    1. Recent price direction: Position relative to EMA 9 and EMA 21, and 5-bar price change.
    2. Momentum: RSI(14) level and recent rate of change.
    3. Volume activity: Volume relative to 20-bar baseline.
    4. Volatility: ATR(14) in points and percentage.
    
    Outputs one of:
    - 'upward scenario'
    - 'downward scenario'
    - 'no clear setup'
    
    Returns a comprehensive dictionary with all factors, thresholds, and transparent rationale.
    """
    if df.empty or len(df) < 5:
        return {
            "scenario": "no clear setup",
            "reason": "Insufficient candle data for meaningful indicator calculation (need at least 5 candles).",
            "factors": {}
        }

    # Ensure indicators exist
    if 'ema_9' not in df.columns or 'rsi' not in df.columns:
        df = compute_indicators(df)

    # Normalize index
    idx = candle_idx if candle_idx >= 0 else len(df) + candle_idx
    if idx < 0 or idx >= len(df):
        idx = len(df) - 1

    row = df.iloc[idx]
    
    close_price = float(row['close'])
    ema9_val = float(row['ema_9'])
    ema21_val = float(row['ema_21'])
    rsi_val = float(row['rsi'])
    vol_ratio = float(row['volume_ratio'])
    atr_val = float(row['atr_14'])
    atr_pct = float(row['atr_pct'])
    ts_str = str(row['timestamp'])

    # 5-bar price change
    lookback_idx = max(0, idx - 5)
    past_close = float(df.iloc[lookback_idx]['close'])
    change_5bar = close_price - past_close
    change_5bar_pct = (change_5bar / past_close) * 100 if past_close > 0 else 0.0

    # 1. Price Direction Evaluation
    direction_status = "Neutral / Mixed"
    dir_score_bull = 0.0
    dir_score_bear = 0.0

    if close_price > ema9_val and ema9_val > ema21_val:
        direction_status = "Bullish Alignment (Price > EMA 9 > EMA 21)"
        dir_score_bull += 1.5
    elif close_price < ema9_val and ema9_val < ema21_val:
        direction_status = "Bearish Alignment (Price < EMA 9 < EMA 21)"
        dir_score_bear += 1.5
    elif close_price > ema9_val:
        direction_status = "Mildly Bullish (Price > EMA 9, below or near EMA 21)"
        dir_score_bull += 0.5
    elif close_price < ema9_val:
        direction_status = "Mildly Bearish (Price < EMA 9, above or near EMA 21)"
        dir_score_bear += 0.5

    if change_5bar > 0:
        dir_score_bull += 0.5
    elif change_5bar < 0:
        dir_score_bear += 0.5

    # 2. Momentum Evaluation (RSI 14)
    mom_status = "Neutral Momentum"
    mom_score_bull = 0.0
    mom_score_bear = 0.0

    if 52.0 <= rsi_val <= 68.0:
        mom_status = f"Healthy Bullish Momentum (RSI: {rsi_val:.1f})"
        mom_score_bull += 1.5
    elif 32.0 <= rsi_val <= 48.0:
        mom_status = f"Healthy Bearish Momentum (RSI: {rsi_val:.1f})"
        mom_score_bear += 1.5
    elif rsi_val > 68.0:
        mom_status = f"Overbought Territory (RSI: {rsi_val:.1f}) - Exhaustion Risk"
        # Overbought reduces pure follow-through confidence
        mom_score_bull += 0.5
    elif rsi_val < 32.0:
        mom_status = f"Oversold Territory (RSI: {rsi_val:.1f}) - Bounce Risk"
        mom_score_bear += 0.5
    else:
        mom_status = f"Sideways / Indecisive Range (RSI: {rsi_val:.1f})"

    # 3. Volume Comparison Evaluation
    vol_status = "Average Volume"
    vol_confirm_bull = False
    vol_confirm_bear = False

    if vol_ratio >= 1.25:
        vol_status = f"Elevated Volume ({vol_ratio:.2f}x 20-bar avg)"
        if dir_score_bull > dir_score_bear:
            vol_confirm_bull = True
        elif dir_score_bear > dir_score_bull:
            vol_confirm_bear = True
    elif vol_ratio <= 0.75:
        vol_status = f"Subdued Volume ({vol_ratio:.2f}x 20-bar avg)"
    else:
        vol_status = f"Typical Volume ({vol_ratio:.2f}x 20-bar avg)"

    # 4. Volatility Evaluation (ATR)
    volatility_status = f"ATR(14): {atr_val:.1f} pts ({atr_pct:.3f}%)"
    if atr_pct > 0.06:
        volatility_status += " - High Volatility"
    elif atr_pct < 0.02:
        volatility_status += " - Compressed Volatility"
    else:
        volatility_status += " - Normal Volatility"

    # Aggregated Rule-based Scoring
    total_bull = dir_score_bull + mom_score_bull + (0.5 if vol_confirm_bull else 0.0)
    total_bear = dir_score_bear + mom_score_bear + (0.5 if vol_confirm_bear else 0.0)

    # Determine Scenario (transparent rule)
    factors_list = []
    
    if total_bull >= 2.5 and total_bull > (total_bear + 1.0):
        scenario = "upward scenario"
        factors_list.append(f"Price is aligned upward ({direction_status})")
        factors_list.append(f"Momentum supports buyers ({mom_status})")
        factors_list.append(f"Volume state: {vol_status}")
        rationale = "Bullish trend structure and positive momentum align with acceptable volume activity."
    elif total_bear >= 2.5 and total_bear > (total_bull + 1.0):
        scenario = "downward scenario"
        factors_list.append(f"Price is aligned downward ({direction_status})")
        factors_list.append(f"Momentum supports sellers ({mom_status})")
        factors_list.append(f"Volume state: {vol_status}")
        rationale = "Bearish trend structure and negative momentum align with selling pressure."
    else:
        scenario = "no clear setup"
        factors_list.append(f"Price Direction: {direction_status}")
        factors_list.append(f"Momentum: {mom_status}")
        factors_list.append(f"Volume: {vol_status}")
        factors_list.append(f"Volatility: {volatility_status}")
        rationale = "Signals are mixed, range-bound, or lacking decisive trend and volume confirmation."

    return {
        "scenario": scenario,
        "candle_timestamp": ts_str,
        "current_price": close_price,
        "price_direction": direction_status,
        "momentum": mom_status,
        "rsi": round(rsi_val, 1),
        "ema_9": round(ema9_val, 2),
        "ema_21": round(ema21_val, 2),
        "volume_status": vol_status,
        "volume_ratio": round(vol_ratio, 2),
        "volatility_status": volatility_status,
        "atr_value": round(atr_val, 2),
        "atr_pct": round(atr_pct, 4),
        "change_5bar": round(change_5bar, 2),
        "factors": factors_list,
        "rationale": rationale,
        "bullish_score": round(total_bull, 1),
        "bearish_score": round(total_bear, 1)
    }
