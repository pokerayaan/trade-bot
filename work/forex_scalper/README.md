# Forex Scalper Bot

Python forex scalping bot with:

- Dynamic "top 6 movers" selection by recent ATR percentage.
- Liquid FX universe focused on major pairs and active crosses.
- 5-minute EMA trend/pullback strategy with RSI, ADX, ATR and spread filters.
- 1% risk per trade by default.
- ATR/swing-based stop, 1.5R take-profit, break-even/trailing management hooks.
- Backtest mode using Yahoo Finance FX data.
- MetaTrader 5 live/paper scan mode, defaulting to `dry_run`.

This is not financial advice. Scalping is execution-sensitive: spreads, slippage, broker rules, latency, news events, and psychology can turn a good-looking backtest into a losing live system. Forward test on demo first.

## Why This Strategy

The internet research used for the design points to the same themes:

- Scalping works best in liquid, active markets and on short timeframes.
- Major pairs such as EUR/USD and USD/JPY are usually preferred for liquidity.
- London/New York overlap is commonly the most active window.
- Technical systems often combine trend filters, momentum confirmation, quick exits, tight stops, and strict risk management.

The bot therefore ranks pairs by recent ATR%, trades only the top movers, then uses:

- `EMA 200` for trend direction.
- `EMA 9/21` for short-term alignment.
- `RSI` for pullback/momentum confirmation.
- `ADX` to avoid dead chop.
- `ATR` and recent swing levels for adaptive stops.
- `1.5R` reward/risk by default, because scalping usually benefits from realistic targets rather than far-away fantasy targets.

## Install

```powershell
cd C:\Users\agadd\Documents\Codex\2026-06-10\hey\work\forex_scalper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`MetaTrader5` only works on Windows with a local MT5 terminal installed.

## Backtest

```powershell
python forex_scalper_bot.py --mode backtest --config config.yaml
```

## Scan Current Signals

```powershell
python forex_scalper_bot.py --mode scan --config config.yaml
```

This downloads recent data, selects the current top 6 movers, and prints signals.

## MetaTrader 5 Dry Run

Keep `execution.mode: dry_run` in `config.yaml`.

```powershell
python forex_scalper_bot.py --mode live-mt5 --config config.yaml
```

## Live Trading

Only after demo testing:

1. Set `execution.mode: live` in `config.yaml`.
2. Confirm symbol names match your broker. Some brokers use suffixes such as `EURUSD.a`.
3. Log into MT5, or provide `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER`.
4. Run:

```powershell
python forex_scalper_bot.py --mode live-mt5 --config config.yaml
```

## Practical Rules Before Going Live

- Demo trade at least 100 signals.
- Avoid high-impact news windows.
- Stop for the day after `max_daily_loss`.
- Keep max open trades small.
- Watch actual spread and slippage per pair.
- Re-backtest whenever changing broker, timeframe, or risk settings.
