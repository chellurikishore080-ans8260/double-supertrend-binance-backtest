from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Double SuperTrend Backtest", layout="wide")

# Fixed requested Binance Spot USDT pairs. MATIC migrated to POL; Binance symbol is POLUSDT.
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "LINKUSDT", "XAUTUSDT", "BCHUSDT", "POLUSDT", "BNBUSDT"]
API_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
INTERVALS = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]


def binance_get(path: str, params: dict):
    """Public spot API request with host fallbacks for regional HTTP 451/availability errors."""
    failures = []
    for host in API_HOSTS:
        try:
            response = requests.get(host + path, params=params, timeout=15)
            if response.ok:
                return response.json()
            failures.append(f"{host.rsplit('/', 1)[-1]} HTTP {response.status_code}: {response.text[:120]}")
        except requests.RequestException as exc:
            failures.append(f"{host}: {exc}")
    raise RuntimeError("All Binance data hosts failed. " + " | ".join(failures))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch the selected date range from Binance Spot klines, paging at Binance's 1000-row limit."""
    rows = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = binance_get("/api/v3/klines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + 1
        if next_cursor <= cursor or len(batch) < 1000:
            break
        cursor = next_cursor
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "close_time"])
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    frame = frame[["timestamp", "open", "high", "low", "close", "volume", "close_time"]].copy()
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="ms", utc=True)
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Never use an in-progress candle in a backtest.
    frame = frame[frame["close_time"] < int(datetime.now(timezone.utc).timestamp() * 1000)]
    return frame.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def supertrend(frame: pd.DataFrame, factor: float, period: int):
    """Pine ta.supertrend direction convention: -1 is uptrend, +1 is downtrend."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(frame) >= period:
        # Wilder RMA with the same initial SMA seed used by ta.atr.
        atr.iloc[period - 1] = tr.iloc[:period].mean()
        for i in range(period, len(frame)):
            atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    mid = (high + low) / 2
    basic_upper, basic_lower = mid + factor * atr, mid - factor * atr
    upper, lower = basic_upper.copy(), basic_lower.copy()
    line = pd.Series(np.nan, index=frame.index, dtype=float)
    direction = pd.Series(np.nan, index=frame.index, dtype=float)
    for i in range(len(frame)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(direction.iloc[i - 1]):
            direction.iloc[i] = 1
            line.iloc[i] = upper.iloc[i]
            continue
        old_upper, old_lower = upper.iloc[i - 1], lower.iloc[i - 1]
        upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < old_upper or close.iloc[i - 1] > old_upper else old_upper
        lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > old_lower or close.iloc[i - 1] < old_lower else old_lower
        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if close.iloc[i] > upper.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < lower.iloc[i] else -1
        line.iloc[i] = lower.iloc[i] if direction.iloc[i] == -1 else upper.iloc[i]
    return line, direction


def make_signals(candles: pd.DataFrame, fast_atr: int, fast_factor: float,
                 slow_atr: int, slow_factor: float, entry_rule: str, exit_rule: str) -> pd.DataFrame:
    df = candles.copy().reset_index(drop=True)
    df["fast_line"], df["fast_dir"] = supertrend(df, fast_factor, fast_atr)
    df["slow_line"], df["slow_dir"] = supertrend(df, slow_factor, slow_atr)
    fast_up, fast_down = df["fast_dir"].eq(-1), df["fast_dir"].eq(1)
    slow_up, slow_down = df["slow_dir"].eq(-1), df["slow_dir"].eq(1)
    both_up, both_down = fast_up & slow_up, fast_down & slow_down
    if entry_rule == "Both agree":
        # Match the Pine rule: signal only on the candle both trends first agree.
        long_entry = both_up & ~both_up.shift(1, fill_value=False)
        short_entry = both_down & ~both_down.shift(1, fill_value=False)
    else:
        # Pine ta.crossover/ta.crossunder of the two plotted SuperTrend values.
        long_entry = df["fast_line"].gt(df["slow_line"]) & df["fast_line"].shift(1).le(df["slow_line"].shift(1)) & fast_up
        short_entry = df["fast_line"].lt(df["slow_line"]) & df["fast_line"].shift(1).ge(df["slow_line"].shift(1)) & fast_down
    df["long_signal"] = long_entry.fillna(False)
    df["short_signal"] = short_entry.fillna(False)
    df["exit_long"] = (fast_down if exit_rule == "Fast flips" else slow_down).fillna(False)
    df["exit_short"] = (fast_up if exit_rule == "Fast flips" else slow_up).fillna(False)
    return df


def backtest_one(symbol: str, candles: pd.DataFrame, start_date: date, end_date: date,
                 notional: float, fee_rate: float, fast_atr: int, fast_factor: float,
                 slow_atr: int, slow_factor: float, entry_rule: str, exit_rule: str,
                 allow_shorts: bool) -> list[dict]:
    df = make_signals(candles, fast_atr, fast_factor, slow_atr, slow_factor, entry_rule, exit_rule)
    start = pd.Timestamp(start_date, tz="UTC")
    end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    position: Optional[dict] = None
    pending: Optional[str] = None
    trades = []
    for row in df.itertuples(index=False):
        # A signal is confirmed at candle close and filled at the following candle's open.
        if pending is not None:
            if position is not None:
                exit_price = float(row.open)
                quantity = notional / position["entry_price"]
                gross = (exit_price - position["entry_price"]) * quantity * position["side"]
                fees = fee_rate * (notional + quantity * exit_price)
                trades.append({"symbol": symbol, "side": "Long" if position["side"] == 1 else "Short",
                    "entry_time": position["entry_time"], "exit_time": row.timestamp,
                    "entry_price": position["entry_price"], "exit_price": exit_price,
                    "notional_usd": notional, "gross_pnl_usd": gross, "fees_usd": fees,
                    "net_pnl_usd": gross - fees, "return_pct": (gross - fees) / notional * 100})
                position = None
            if pending in ("long", "short") and (pending == "long" or allow_shorts):
                position = {"side": 1 if pending == "long" else -1,
                            "entry_price": float(row.open), "entry_time": row.timestamp}
            pending = None
        if row.timestamp < start or row.timestamp >= end_exclusive:
            continue
        long_signal = bool(row.long_signal)
        short_signal = bool(row.short_signal) and allow_shorts
        if position is None:
            if long_signal:
                pending = "long"
            elif short_signal:
                pending = "short"
        elif position["side"] == 1 and (short_signal or bool(row.exit_long)):
            pending = "short" if short_signal else "exit"
        elif position["side"] == -1 and (long_signal or bool(row.exit_short)):
            pending = "long" if long_signal else "exit"
    eligible = df[df["timestamp"] < end_exclusive]
    if position is not None and not eligible.empty:
        last = eligible.iloc[-1]
        exit_price = float(last["close"])
        quantity = notional / position["entry_price"]
        gross = (exit_price - position["entry_price"]) * quantity * position["side"]
        fees = fee_rate * (notional + quantity * exit_price)
        trades.append({"symbol": symbol, "side": "Long" if position["side"] == 1 else "Short",
            "entry_time": position["entry_time"], "exit_time": last["timestamp"],
            "entry_price": position["entry_price"], "exit_price": exit_price,
            "notional_usd": notional, "gross_pnl_usd": gross, "fees_usd": fees,
            "net_pnl_usd": gross - fees, "return_pct": (gross - fees) / notional * 100,
            "note": "Open position marked to final close"})
    return trades


def performance(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"Trades": 0, "Win rate": 0.0, "Net P&L ($)": 0.0, "Profit factor": 0.0, "Max drawdown ($)": 0.0}
    pnl = trades.sort_values(["exit_time", "symbol"])["net_pnl_usd"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    return {"Trades": len(pnl), "Win rate": (pnl > 0).mean() * 100,
        "Net P&L ($)": pnl.sum(),
        "Profit factor": wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf"),
        "Max drawdown ($)": (equity.cummax() - equity).max(),
        "Average trade ($)": pnl.mean()}


st.title("Double SuperTrend — Binance Spot Backtest")
st.caption("Requested USDT pairs only • Binance Spot klines • $1,000 notional per trade • no live orders")
st.markdown("**Signal logic:** Pine Double SuperTrend defaults: Fast ATR 10 × 1.5, Slow ATR 10 × 3.0. Long entry when both SuperTrends first agree bullish; default exit when Fast SuperTrend flips bearish. Signals use closed candles and fill at the next candle open. Spot mode is long-only by default.")

with st.sidebar:
    st.header("Backtest settings")
    timeframe = st.selectbox("Timeframe", INTERVALS, index=3, help="Default is 1-hour candles. Choose the candle interval used for both SuperTrends.")
    start_date = st.date_input("Backtest start (UTC)", value=date(2024, 1, 1))
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    end_date = st.date_input("Backtest end (UTC)", value=yesterday, max_value=yesterday)
    st.caption("Default period: 1 Jan 2024 through yesterday. Change dates here for another historical period.")
    notional = st.number_input("Notional per trade (USD)", min_value=10.0, value=1000.0, step=100.0)
    fee_pct = st.number_input("Fee per side (%)", min_value=0.0, value=0.1, step=0.01, format="%.3f")
    st.subheader("Pine strategy parameters")
    fast_atr = st.number_input("Fast ATR length", min_value=1, value=10, step=1)
    fast_factor = st.number_input("Fast factor", min_value=0.1, value=1.5, step=0.1)
    slow_atr = st.number_input("Slow ATR length", min_value=1, value=10, step=1)
    slow_factor = st.number_input("Slow factor", min_value=0.1, value=3.0, step=0.1)
    entry_rule = st.selectbox("Entry rule", ["Both agree", "Fast line crosses slow line"])
    exit_rule = st.selectbox("Exit rule", ["Fast flips", "Slow flips"])
    allow_shorts = st.checkbox("Include shorts (not ordinary spot trading)", value=False)

st.write("**Pairs:** " + " · ".join(SYMBOLS) + " (MATIC is now represented by Binance pair POLUSDT where available).")
st.info("Backtest uses Binance Spot `/api/v3/klines` with market-data host fallbacks. Some requested pairs may not be listed as USDT spot markets in your region; those are reported individually as skipped.")
if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

if st.button("Run Double SuperTrend Backtest", type="primary"):
    start_ms = int(pd.Timestamp(start_date - timedelta(days=90), tz="UTC").timestamp() * 1000)
    # Binance endTime is inclusive; include the entire selected end date.
    end_ms = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    all_trades, warnings, loaded = [], [], []
    progress, status = st.progress(0), st.empty()
    for i, symbol in enumerate(SYMBOLS, start=1):
        status.text(f"{i}/{len(SYMBOLS)} — {symbol} {timeframe} candles load chestunnanu")
        try:
            candles = fetch_klines(symbol, timeframe, start_ms, end_ms)
            if candles.empty:
                warnings.append(f"{symbol}: Binance returned no candles for this date range (pair may not be listed as USDT spot).")
            else:
                loaded.append(f"{symbol}: {len(candles):,} candles")
                all_trades.extend(backtest_one(symbol, candles, start_date, end_date, notional,
                    fee_pct / 100, int(fast_atr), float(fast_factor), int(slow_atr),
                    float(slow_factor), entry_rule, exit_rule, allow_shorts))
        except Exception as exc:
            warnings.append(f"{symbol}: {exc}")
        progress.progress(i / len(SYMBOLS))
    status.empty()
    progress.empty()
    st.session_state["bt_trades"] = pd.DataFrame(all_trades)
    st.session_state["bt_warnings"] = warnings
    st.session_state["bt_loaded"] = loaded
    st.session_state["bt_config"] = {"timeframe": timeframe, "start": str(start_date), "end": str(end_date), "notional": notional, "entry": entry_rule, "exit": exit_rule}

if "bt_trades" in st.session_state:
    config = st.session_state["bt_config"]
    st.caption(f"Last run: {config['timeframe']} candles | {config['start']} to {config['end']} UTC | ${config['notional']:,.0f} notional per entry | Entry: {config['entry']} | Exit: {config['exit']}")
    if st.session_state["bt_loaded"]:
        st.write("**Data loaded:** " + " · ".join(st.session_state["bt_loaded"]))
    warnings = st.session_state["bt_warnings"]
    if warnings:
        with st.expander(f"Skipped pairs / data warnings ({len(warnings)})"):
            st.write("\n".join(warnings))
    trades = st.session_state["bt_trades"]
    if trades.empty:
        st.warning("No trades generated with these exact entry/exit rules in this date range. Confirm candle counts above. Try a longer date range or choose 'Fast line crosses slow line'; signal rules tirigi marchadam kaadu—change cheste results marutayi.")
    else:
        stats = performance(trades)
        cols = st.columns(5)
        cols[0].metric("Trades", f"{stats['Trades']:,}")
        cols[1].metric("Win rate", f"{stats['Win rate']:.2f}%")
        cols[2].metric("Net P&L", f"${stats['Net P&L ($)']:,.2f}")
        cols[3].metric("Profit factor", "∞" if np.isinf(stats["Profit factor"]) else f"{stats['Profit factor']:.2f}")
        cols[4].metric("Max drawdown", f"${stats['Max drawdown ($)']:,.2f}")
        st.subheader("Cumulative realized P&L")
        curve = trades.sort_values(["exit_time", "symbol"]).set_index("exit_time")["net_pnl_usd"].cumsum().rename("P&L ($)")
        st.line_chart(curve)
        st.subheader("Pair-wise results")
        per_pair = trades.groupby("symbol").agg(trades=("net_pnl_usd", "size"), net_pnl_usd=("net_pnl_usd", "sum"), win_rate_pct=("net_pnl_usd", lambda x: (x > 0).mean() * 100)).sort_values("net_pnl_usd", ascending=False)
        st.dataframe(per_pair, use_container_width=True)
        st.subheader("Trade log")
        st.dataframe(trades.sort_values("entry_time", ascending=False), use_container_width=True)
        st.download_button("Download trade log CSV", trades.to_csv(index=False), "double_supertrend_trades.csv", "text/csv")
        st.caption("Each pair may hold its own $1,000 position concurrently; 9 simultaneous positions require up to $9,000. Fees are modeled both sides. Slippage/tax/lot-size rules are not modeled. Hypothetical results, not investment advice.")
