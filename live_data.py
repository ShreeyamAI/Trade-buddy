"""
Live Market Data Module for Trade Buddy.
Fetches real-time NIFTY 50 1-minute candles via yfinance (Yahoo Finance).
Detects Indian Standard Time (IST) market session status.

IMPORTANT NOTICES:
- Data is sourced from Yahoo Finance via yfinance. Yahoo Finance data may be delayed
  (typically ~15 min for indices) and is NOT licensed for commercial use.
- Before using in production, check Yahoo Finance's Terms of Service independently.
- This module does NOT connect to any broker or execute orders.
"""

import pandas as pd
from datetime import datetime, time, timedelta
from typing import Optional, Tuple, Dict, Any
import pytz

# NIFTY 50 Yahoo Finance ticker
NIFTY50_TICKER = "^NSEI"

# Indian Standard Time
IST = pytz.timezone('Asia/Kolkata')

# NSE market session hours (IST)
MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN     = time(9, 0)      # Pre-open call auction starts
WEEKDAYS     = {0, 1, 2, 3, 4}  # Mon–Fri


def ist_now() -> datetime:
    """Returns the current datetime in IST."""
    return datetime.now(IST)


def market_status() -> Dict[str, Any]:
    """
    Returns a dict describing the current NSE market session status.
    Fields: is_open (bool), phase (str), message (str), time_str (str)
    """
    now = ist_now()
    t = now.time()
    wd = now.weekday()

    time_str = now.strftime('%H:%M:%S IST')
    date_str = now.strftime('%d %b %Y')

    if wd not in WEEKDAYS:
        return {"is_open": False, "phase": "Weekend",
                "message": f"Market closed — {now.strftime('%A')}",
                "time_str": time_str, "date_str": date_str}

    if t < PRE_OPEN:
        mins = int((datetime.combine(now.date(), PRE_OPEN) -
                    datetime.combine(now.date(), t)).seconds / 60)
        return {"is_open": False, "phase": "Pre-market",
                "message": f"Pre-open auction in ~{mins} min",
                "time_str": time_str, "date_str": date_str}

    if PRE_OPEN <= t < MARKET_OPEN:
        return {"is_open": False, "phase": "Pre-open",
                "message": "Pre-open call auction in progress (09:00–09:15)",
                "time_str": time_str, "date_str": date_str}

    if MARKET_OPEN <= t <= MARKET_CLOSE:
        left = int((datetime.combine(now.date(), MARKET_CLOSE) -
                    datetime.combine(now.date(), t)).seconds / 60)
        return {"is_open": True, "phase": "Market Open",
                "message": f"🟢 Market Open — {left} min remaining (closes 15:30)",
                "time_str": time_str, "date_str": date_str}

    return {"is_open": False, "phase": "After Hours",
            "message": "Market closed — After hours (session ended 15:30)",
            "time_str": time_str, "date_str": date_str}


def fetch_live_nifty(interval: str = "1m", period: str = "1d") -> Tuple[Optional[pd.DataFrame], str]:
    """
    Downloads NIFTY 50 candle data using yfinance.
    
    Parameters:
        interval: Bar interval ('1m', '2m', '5m') — 1m data available for last 7 days only.
        period: Lookback period ('1d', '2d', '5d').
    
    Returns:
        (DataFrame | None, status_message)
    The DataFrame uses Trade Buddy's standard columns:
        timestamp, open, high, low, close, volume
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(NIFTY50_TICKER)
        raw = ticker.history(period=period, interval=interval, auto_adjust=True)

        if raw is None or raw.empty:
            return None, "No data returned. Market may be closed or Yahoo Finance unavailable."

        raw = raw.reset_index()

        # Column name normalisation (yfinance v0.2+ uses 'Datetime' for intraday)
        ts_col = 'Datetime' if 'Datetime' in raw.columns else 'Date'
        raw = raw.rename(columns={
            ts_col: 'timestamp',
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume',
        })

        # Convert timezone-aware timestamps → IST → naive
        if hasattr(raw['timestamp'].dtype, 'tz') or str(raw['timestamp'].dtype) == 'object':
            raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
        else:
            raw['timestamp'] = pd.to_datetime(raw['timestamp'])

        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in raw.columns]
        if missing:
            return None, f"Missing columns from yfinance response: {missing}"

        df = raw[required].dropna(subset=['open', 'high', 'low', 'close']).copy()
        df = df.reset_index(drop=True)

        # Round timestamp to remove microseconds
        df['timestamp'] = df['timestamp'].dt.round('min')

        last_ts = str(df['timestamp'].iloc[-1])
        status = (
            f"✅ Live data loaded: {len(df)} bars | "
            f"Last candle: {last_ts} | "
            f"Source: Yahoo Finance (^NSEI) | "
            f"⚠️ May be delayed up to 15 min"
        )
        return df, status

    except ImportError:
        return None, "yfinance is not installed. Run: pip install yfinance"
    except Exception as e:
        err = str(e)
        if "No data found" in err or "404" in err:
            return None, "No NIFTY 50 data available. Yahoo Finance may be down or the market is closed."
        return None, f"Data fetch error: {err[:150]}"
