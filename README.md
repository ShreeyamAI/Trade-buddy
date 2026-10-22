# Trade Buddy

**Trade Buddy** is a local, analysis-only dashboard for studying short-term price movements in the **NIFTY 50 index**.

> ⚠️ **Strict Safety Notice**: Trade Buddy is strictly an educational and analytical research environment. It **never places trades**, does **not connect to any broker**, and contains **no order execution or broker credentials**. All evaluations are paper-tracking historical scenarios, not financial advice or profit guarantees.

---

## Tech Stack

- **Python 3.9+**
- **Streamlit**: Interactive web dashboard interface
- **Pandas**: Fast, validated data handling and rolling calculations
- **Plotly**: Multi-panel interactive candlestick and volume charting
- **SQLite**: Lightweight local database for scenario logging and paper-tracking verification

---

## Project Structure

```
Trade Buddy/
├── app.py                     # Main Streamlit dashboard application
├── data_loader.py             # CSV loading and strict data integrity validation
├── analysis.py                # Technical analysis and rule-based scenario engine
├── charts.py                  # Plotly interactive candlestick and volume charts
├── database.py                # SQLite logging and evaluation tracking
├── requirements.txt           # Python package dependencies
├── trade_buddy.db             # Local SQLite database (auto-created upon first run)
├── data/
│   └── sample_nifty50_1min.csv# Verified sample 1-minute NIFTY 50 dataset
└── README.md                  # Documentation and setup guide
```

---

## Setup & Run Instructions (macOS)

### 1. Open Terminal and Navigate to the Project

```bash
cd "/Volumes/shreeyam/Projects/Trade Buddy"
```

### 2. Set Up a Virtual Environment

```bash
# Create a virtual environment (if not already created)
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Streamlit will open the app in your default browser at `http://localhost:8501`.

---

## CSV Data Format

To use your own historical data with Trade Buddy, save your candles as a CSV file with the following standard columns (case-insensitive):

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | String / Datetime | `2024-10-10 09:15:00` | 1-minute candle timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `open` | Float | `24980.00` | Opening index level |
| `high` | Float | `24995.50` | Highest index level in the minute |
| `low` | Float | `24976.25` | Lowest index level in the minute |
| `close` | Float | `24991.10` | Closing index level |
| `volume` | Integer | `35200` | Total traded volume/contracts in the minute |

### Example CSV Snippet

```csv
timestamp,open,high,low,close,volume
2024-10-10 09:15:00,24980.00,24995.50,24976.25,24991.10,35200
2024-10-10 09:16:00,24991.10,25002.40,24988.30,24998.60,28400
2024-10-10 09:17:00,24998.60,25005.10,24994.00,25001.20,24100
```

### Data Validation Checks

Trade Buddy validates the dataset immediately upon loading:
1. **Required Columns**: Confirms all 6 mandatory columns exist.
2. **Missing Values**: Rejects files with empty or NaN values.
3. **Timestamps**: Ensures dates are parseable into valid datetime objects.
4. **Duplicates**: Rejects duplicate timestamps.
5. **Chronological Order**: Enforces strictly increasing ascending timestamps.
6. **Price Sanity**: Confirms $High \ge \max(Open, Close)$, $Low \le \min(Open, Close)$, positive prices, and non-negative volume.

---

## Replacing the Sample Data with a Real Data Provider

Trade Buddy comes with `sample_nifty50_1min.csv` so you can immediately explore the dashboard without external accounts.

To study actual historical sessions:
1. Obtain historical 1-minute NIFTY 50 OHLCV data from an authorized, legitimate data vendor (such as **Zerodha Kite Connect Historical Data API**, **Upstox Historical API**, **TrueData Market Data**, **Global Datafeeds**, or **NSE India Archives**).
2. Format the export into the 6 standard columns described above.
3. Select **"Upload Custom CSV"** in the sidebar, or replace `data/sample_nifty50_1min.csv` with your file.

> ⚠️ **Important Data Provider Advisory**:
> - Trade Buddy does not scrape websites or assume unauthenticated market data feeds.
> - Before connecting or purchasing market data, check the provider's **service availability**, **pricing plans**, **licensing terms**, and **API usage restrictions**.

---

## How the Scenario Assessment Works

Trade Buddy evaluates four objective technical inputs for the chosen candle:
1. **Recent Price Direction**: Position of index price relative to 9-period EMA and 21-period EMA, combined with 5-candle net price change.
2. **Momentum**: 14-period Relative Strength Index (RSI) to detect bullish (52–68), bearish (32–48), or exhausted/neutral conditions.
3. **Volume Comparison**: Compares the candle volume with the 20-period Simple Moving Average volume ratio to check whether volume confirms the move.
4. **Short-Term Volatility**: 14-period Average True Range (ATR) in points and percentage of index price.

Based on these transparent rules, it displays one of:
- **`upward scenario`**: Bullish trend alignment and healthy positive momentum with volume support.
- **`downward scenario`**: Bearish trend alignment and negative momentum with selling pressure.
- **`no clear setup`**: Mixed signals, consolidation, low volume churn, or extreme exhaustion.

### Horizon Evaluation & Paper Tracking

- Choose an evaluation horizon of **5, 10, or 15 minutes**.
- Click **"Log This Scenario for Paper Tracking"** to record the snapshot to SQLite.
- As subsequent candle data is loaded, Trade Buddy checks the index price at the target horizon timestamp and categorizes the observed move as **ALIGNED**, **OPPOSITE**, or **NEUTRAL**.
- Summary statistics display the directional alignment rate and average point movement.
