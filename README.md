# Double SuperTrend Binance Spot Backtest

A Streamlit dashboard to backtest the Tanmay Kurtkoti Double SuperTrend strategy on Binance Vision spot candles.

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Select **Create app**.
3. Repository: `chellurikishore080-ans8260/double-supertrend-binance-backtest`
4. Branch: `main`
5. Main file path: `app.py`
6. Select **Deploy**. Streamlit installs the dependencies from `requirements.txt`.

No Binance API keys are needed. This app reads public market data only and does not place live trades.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Backtest notes

- Selects active spot USDT pairs using current 24-hour quote volume (up to 50). Applying today's ranking to history causes survivorship bias; historical membership is not reconstructed.
- Signals use completed candles and are intended to execute at the next candle open.
- The default is long-only, with a configurable fixed USD notional per entry ($1,000 by default) and estimated fees on entry/exit.
- Slippage, taxes, minimum order sizes, and exact exchange execution are not modeled. Multiple coins can have positions open at once, each with its own configured notional.
- Results are hypothetical and are not investment advice.
