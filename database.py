"""
Local SQLite Storage and Paper-Tracking Evaluation Module for Trade Buddy.
Logs generated scenarios and verifies them against subsequent observed index moves across
5-, 10-, or 15-minute evaluation horizons.
Strictly labeled as historical paper tracking; never implies real profitability or trade execution.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import os

DB_FILENAME = "trade_buddy.db"


def get_db_path() -> str:
    """Returns absolute path to the SQLite database file in the project directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DB_FILENAME)


def init_db():
    """Initializes SQLite schema if not already present."""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scenario_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                candle_timestamp TEXT NOT NULL,
                candle_close REAL NOT NULL,
                scenario TEXT NOT NULL,
                horizon_minutes INTEGER NOT NULL,
                target_timestamp TEXT NOT NULL,
                factors_json TEXT,
                evaluation_status TEXT DEFAULT 'PENDING',
                observed_timestamp TEXT,
                observed_price REAL,
                price_change REAL,
                price_change_pct REAL,
                outcome TEXT DEFAULT 'PENDING',
                notes TEXT
            );
        """)
        conn.commit()


def log_scenario(
    candle_timestamp: str,
    candle_close: float,
    scenario: str,
    horizon_minutes: int,
    factors: Dict[str, Any],
    notes: str = ""
) -> int:
    """
    Logs an experimental analysis scenario to the SQLite database.
    Calculates target timestamp based on evaluation horizon (5, 10, or 15 minutes).
    """
    init_db()
    db_path = get_db_path()

    # Parse candle timestamp to compute target timestamp
    try:
        ts_dt = pd.to_datetime(candle_timestamp)
        target_dt = ts_dt + timedelta(minutes=int(horizon_minutes))
        target_timestamp_str = target_dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        target_timestamp_str = f"{candle_timestamp} + {horizon_minutes}m"

    logged_at_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    factors_str = json.dumps(factors)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scenario_log (
                logged_at, candle_timestamp, candle_close, scenario,
                horizon_minutes, target_timestamp, factors_json,
                evaluation_status, outcome, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING', ?)
        """, (
            logged_at_str,
            str(candle_timestamp),
            float(candle_close),
            str(scenario),
            int(horizon_minutes),
            target_timestamp_str,
            factors_str,
            notes
        ))
        conn.commit()
        return cursor.lastrowid


def update_evaluations(df: pd.DataFrame) -> int:
    """
    Scans pending scenarios and checks if subsequent candle data in `df`
    has reached or exceeded target_timestamp. If available, records observed price
    and categorizes outcome (ALIGNED, OPPOSITE, or NEUTRAL).
    
    Returns the number of scenarios evaluated and updated.
    """
    init_db()
    if df.empty or 'timestamp' not in df.columns or 'close' not in df.columns:
        return 0

    db_path = get_db_path()
    updated_count = 0

    # Ensure timestamp in df is datetime
    df_temp = df.copy()
    df_temp['dt'] = pd.to_datetime(df_temp['timestamp'])

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, candle_timestamp, candle_close, scenario, horizon_minutes, target_timestamp
            FROM scenario_log
            WHERE evaluation_status = 'PENDING'
        """)
        pending_rows = cursor.fetchall()

        for row in pending_rows:
            target_ts = pd.to_datetime(row['target_timestamp'])
            initial_close = float(row['candle_close'])
            scenario = row['scenario']

            # Find matching candle at or immediately after target timestamp (within 3 min window)
            matching = df_temp[df_temp['dt'] >= target_ts].sort_values(by='dt')
            if not matching.empty:
                first_match = matching.iloc[0]
                observed_dt = first_match['dt']
                
                # Check if observed candle is reasonably close to target
                if (observed_dt - target_ts) <= timedelta(minutes=5):
                    observed_price = float(first_match['close'])
                    price_change = round(observed_price - initial_close, 2)
                    price_change_pct = round((price_change / initial_close) * 100, 4)

                    # Assess paper outcome
                    # A threshold of 1 point (or 0.005%) distinguishes neutral drift
                    if scenario == "upward scenario":
                        if price_change > 0.5:
                            outcome = "ALIGNED (Index Rose)"
                        elif price_change < -0.5:
                            outcome = "OPPOSITE (Index Fell)"
                        else:
                            outcome = "NEUTRAL / FLAT"
                    elif scenario == "downward scenario":
                        if price_change < -0.5:
                            outcome = "ALIGNED (Index Fell)"
                        elif price_change > 0.5:
                            outcome = "OPPOSITE (Index Rose)"
                        else:
                            outcome = "NEUTRAL / FLAT"
                    else:  # 'no clear setup'
                        if abs(price_change) <= 2.0:
                            outcome = "ALIGNED (Remained Range-Bound)"
                        else:
                            outcome = "UNEXPECTED BREAKOUT"

                    cursor.execute("""
                        UPDATE scenario_log
                        SET evaluation_status = 'EVALUATED',
                            observed_timestamp = ?,
                            observed_price = ?,
                            price_change = ?,
                            price_change_pct = ?,
                            outcome = ?
                        WHERE id = ?
                    """, (
                        str(first_match['timestamp']),
                        observed_price,
                        price_change,
                        price_change_pct,
                        outcome,
                        row['id']
                    ))
                    updated_count += 1

        conn.commit()

    return updated_count


def get_all_scenarios() -> pd.DataFrame:
    """Retrieves all logged scenarios as a pandas DataFrame."""
    init_db()
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scenario_log ORDER BY id DESC",
            conn
        )
    return df


def get_summary_statistics() -> Dict[str, Any]:
    """Calculates aggregate paper-tracking summary statistics."""
    df = get_all_scenarios()
    if df.empty:
        return {
            "total_logged": 0,
            "evaluated": 0,
            "pending": 0,
            "aligned_count": 0,
            "opposite_count": 0,
            "alignment_rate_pct": 0.0,
            "avg_move_points": 0.0
        }

    total = len(df)
    evaluated_df = df[df['evaluation_status'] == 'EVALUATED']
    evaluated = len(evaluated_df)
    pending = total - evaluated

    aligned = evaluated_df['outcome'].str.startswith('ALIGNED').sum()
    opposite = evaluated_df['outcome'].str.startswith('OPPOSITE').sum()
    
    directional = aligned + opposite
    alignment_rate = (aligned / directional * 100.0) if directional > 0 else 0.0
    avg_change = evaluated_df['price_change'].mean() if evaluated > 0 else 0.0

    return {
        "total_logged": total,
        "evaluated": evaluated,
        "pending": pending,
        "aligned_count": int(aligned),
        "opposite_count": int(opposite),
        "alignment_rate_pct": float(round(alignment_rate, 1)),
        "avg_move_points": float(round(avg_change, 2)) if pd.notnull(avg_change) else 0.0
    }


def clear_database():
    """Wipes the scenario log table for fresh testing."""
    init_db()
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scenario_log;")
        conn.commit()
