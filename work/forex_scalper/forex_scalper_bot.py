from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import time
from datetime import datetime, time as dtime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yaml


YF_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "EURGBP": "EURGBP=X",
    "EURAUD": "EURAUD=X",
}


PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCHF": 0.0001,
    "EURGBP": 0.0001,
    "EURAUD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "AUDJPY": 0.01,
}


@dataclasses.dataclass
class Signal:
    pair: str
    side: str
    entry: float
    stop: float
    target: float
    risk_pips: float
    units: int
    reason: str


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_yfinance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    df = raw.rename(columns=str.lower)
    needed = ["open", "high", "low", "close"]
    return df[[c for c in needed if c in df.columns]].dropna()


def download_pair(pair: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    import yfinance as yf

    ticker = YF_SYMBOLS[pair]
    raw = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    df = normalize_yfinance_frame(raw)
    if df.empty:
        raise RuntimeError(f"No data returned for {pair} ({ticker})")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_atr = atr(df, length)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / tr_atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / tr_atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    s = cfg["strategy"]
    out = df.copy()
    out["ema_fast"] = ema(out["close"], s["fast_ema"])
    out["ema_slow"] = ema(out["close"], s["slow_ema"])
    out["ema_trend"] = ema(out["close"], s["trend_ema"])
    out["rsi"] = rsi(out["close"], s["rsi_period"])
    out["atr"] = atr(out, s["atr_period"])
    out["adx"] = adx(out, s["atr_period"])
    out["swing_low"] = out["low"].rolling(8).min()
    out["swing_high"] = out["high"].rolling(8).max()
    return out.dropna()


def atr_percent(df: pd.DataFrame, cfg: dict[str, Any]) -> float:
    enriched = add_indicators(df, cfg)
    if enriched.empty:
        return 0.0
    latest = enriched.iloc[-1]
    return float(latest["atr"] / latest["close"])


def select_top_movers(data: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> list[str]:
    scores = [(pair, atr_percent(df, cfg)) for pair, df in data.items()]
    scores.sort(key=lambda item: item[1], reverse=True)
    return [pair for pair, _ in scores[: cfg["strategy"]["top_n_pairs"]]]


def in_session(now: datetime, cfg: dict[str, Any]) -> bool:
    filters = cfg["filters"]
    if now.weekday() not in filters["trade_days_utc"]:
        return False
    if not filters.get("london_new_york_overlap_only", True):
        return True
    start = dtime.fromisoformat(filters["session_start_utc"])
    end = dtime.fromisoformat(filters["session_end_utc"])
    current = now.astimezone(timezone.utc).time()
    return start <= current <= end


def pip_size(pair: str) -> float:
    return PIP_SIZE.get(pair, 0.0001)


def approx_pip_value_usd(pair: str, price: float) -> float:
    # Approximate USD pip value per 1 standard lot. Broker-reported values are preferred for live trading.
    if pair.endswith("USD") and not pair.startswith("USD"):
        return 10.0
    if pair.startswith("USD"):
        return 10.0 / max(price, 1e-9)
    if "JPY" in pair:
        return 9.0
    return 10.0


def position_units(pair: str, entry: float, stop: float, equity: float, risk_fraction: float) -> int:
    risk_cash = equity * risk_fraction
    risk_pips = abs(entry - stop) / pip_size(pair)
    if risk_pips <= 0:
        return 0
    lots = risk_cash / (risk_pips * approx_pip_value_usd(pair, entry))
    return max(0, int(lots * 100_000))


def build_signal(pair: str, df: pd.DataFrame, cfg: dict[str, Any]) -> Signal | None:
    enriched = add_indicators(df, cfg)
    if len(enriched) < 5:
        return None

    s = cfg["strategy"]
    latest = enriched.iloc[-1]
    prev = enriched.iloc[-2]
    entry = float(latest["close"])
    atr_now = float(latest["atr"])
    min_adx = float(s["min_adx"])

    if latest["adx"] < min_adx:
        return None

    long_trend = latest["close"] > latest["ema_trend"] and latest["ema_fast"] > latest["ema_slow"]
    short_trend = latest["close"] < latest["ema_trend"] and latest["ema_fast"] < latest["ema_slow"]
    long_trigger = prev["rsi"] < 50 <= latest["rsi"] and latest["close"] > prev["high"]
    short_trigger = prev["rsi"] > 50 >= latest["rsi"] and latest["close"] < prev["low"]

    if long_trend and long_trigger:
        atr_stop = entry - s["stop_atr_multiple"] * atr_now
        swing_stop = float(latest["swing_low"]) - 0.2 * atr_now
        stop = min(atr_stop, swing_stop)
        target = entry + s["reward_risk"] * (entry - stop)
        side = "buy"
        reason = "EMA trend up, RSI reclaimed 50, price broke prior high"
    elif short_trend and short_trigger:
        atr_stop = entry + s["stop_atr_multiple"] * atr_now
        swing_stop = float(latest["swing_high"]) + 0.2 * atr_now
        stop = max(atr_stop, swing_stop)
        target = entry - s["reward_risk"] * (stop - entry)
        side = "sell"
        reason = "EMA trend down, RSI lost 50, price broke prior low"
    else:
        return None

    risk_pips = abs(entry - stop) / pip_size(pair)
    if risk_pips <= 0 or not math.isfinite(risk_pips):
        return None

    units = position_units(
        pair,
        entry,
        stop,
        float(cfg["account"]["equity"]),
        float(cfg["account"]["risk_per_trade"]),
    )
    if units <= 0:
        return None
    return Signal(pair, side, entry, stop, target, risk_pips, units, reason)


def scan(cfg: dict[str, Any]) -> list[Signal]:
    interval = cfg["strategy"]["timeframe"]
    data = {pair: download_pair(pair, interval=interval) for pair in cfg["universe"]}
    return scan_data(data, cfg)


def scan_data(data: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> list[Signal]:
    top_pairs = select_top_movers(data, cfg)
    print("Top movers by ATR%:", ", ".join(top_pairs))
    signals: list[Signal] = []
    for pair in top_pairs:
        signal = build_signal(pair, data[pair], cfg)
        if signal:
            signals.append(signal)
    return signals


def backtest_pair(pair: str, df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    enriched = add_indicators(df, cfg)
    equity = float(cfg["account"]["equity"])
    risk_fraction = float(cfg["account"]["risk_per_trade"])
    reward_risk = float(cfg["strategy"]["reward_risk"])
    trades = []

    for i in range(250, len(enriched) - 1):
        window = enriched.iloc[: i + 1]
        signal = build_signal(pair, window, cfg)
        if not signal:
            continue

        forward = enriched.iloc[i + 1 : i + 25]
        if forward.empty:
            continue
        outcome_r = 0.0
        exit_time = forward.index[-1]
        for ts, row in forward.iterrows():
            if signal.side == "buy":
                stopped = row["low"] <= signal.stop
                targeted = row["high"] >= signal.target
            else:
                stopped = row["high"] >= signal.stop
                targeted = row["low"] <= signal.target
            if stopped and targeted:
                outcome_r = -1.0
                exit_time = ts
                break
            if stopped:
                outcome_r = -1.0
                exit_time = ts
                break
            if targeted:
                outcome_r = reward_risk
                exit_time = ts
                break
        trades.append({"time": enriched.index[i], "exit_time": exit_time, "side": signal.side, "r": outcome_r})
        equity += equity * risk_fraction * outcome_r

    wins = [t for t in trades if t["r"] > 0]
    losses = [t for t in trades if t["r"] < 0]
    return {
        "pair": pair,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "avg_r": float(np.mean([t["r"] for t in trades])) if trades else 0,
        "ending_equity": equity,
        "wins": len(wins),
        "losses": len(losses),
    }


def backtest(cfg: dict[str, Any]) -> None:
    interval = cfg["strategy"]["timeframe"]
    data = {pair: download_pair(pair, interval=interval) for pair in cfg["universe"]}
    top_pairs = select_top_movers(data, cfg)
    print("Backtesting current top movers:", ", ".join(top_pairs))
    results = [backtest_pair(pair, data[pair], cfg) for pair in top_pairs]
    summary = pd.DataFrame(results).sort_values("ending_equity", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


def format_signal(signal: Signal) -> str:
    return (
        f"{signal.pair} {signal.side.upper()} entry={signal.entry:.5f} "
        f"stop={signal.stop:.5f} target={signal.target:.5f} "
        f"risk={signal.risk_pips:.1f} pips units={signal.units} | {signal.reason}"
    )


def live_mt5(cfg: dict[str, Any]) -> None:
    import MetaTrader5 as mt5

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    if login and password and server:
        ok = mt5.initialize(login=int(login), password=password, server=server)
    else:
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    dry_run = cfg["execution"]["mode"] != "live"
    print(f"MT5 initialized. mode={'dry_run' if dry_run else 'live'}")
    try:
        while True:
            now = datetime.now(timezone.utc)
            if not in_session(now, cfg):
                print(f"{now.isoformat()} outside configured trading session")
                time.sleep(int(cfg["execution"]["poll_seconds"]))
                continue
            data = mt5_market_data(mt5, cfg)
            signals = scan_data(data, cfg)
            for signal in signals:
                print(format_signal(signal))
                if spread_too_wide(mt5, signal.pair, cfg):
                    print(f"Skipping {signal.pair}: spread is wider than configured limit")
                    continue
                if dry_run:
                    continue
                place_mt5_order(signal, cfg, mt5)
            time.sleep(int(cfg["execution"]["poll_seconds"]))
    finally:
        mt5.shutdown()


def mt5_market_data(mt5: Any, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    timeframe = mt5.TIMEFRAME_M5
    if cfg["strategy"]["timeframe"] != "5m":
        raise ValueError("MT5 mode currently supports timeframe: 5m")
    bars = int(cfg["strategy"]["lookback_bars"])
    data = {}
    for pair in cfg["universe"]:
        if mt5.symbol_info(pair) is None:
            print(f"Skipping {pair}: symbol not found in MT5")
            continue
        rates = mt5.copy_rates_from_pos(pair, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            print(f"Skipping {pair}: no MT5 rates")
            continue
        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("time")
        data[pair] = frame.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})[
            ["open", "high", "low", "close"]
        ]
    if not data:
        raise RuntimeError("No MT5 market data available. Check broker symbol names.")
    return data


def spread_too_wide(mt5: Any, pair: str, cfg: dict[str, Any]) -> bool:
    limit = cfg["filters"]["max_spread_pips"].get(pair)
    if limit is None:
        return False
    tick = mt5.symbol_info_tick(pair)
    if tick is None:
        return True
    spread_pips = abs(tick.ask - tick.bid) / pip_size(pair)
    return spread_pips > float(limit)


def place_mt5_order(signal: Signal, cfg: dict[str, Any], mt5: Any) -> None:
    symbol_info = mt5.symbol_info(signal.pair)
    if symbol_info is None:
        raise RuntimeError(f"Symbol not found in MT5: {signal.pair}")
    if not symbol_info.visible:
        mt5.symbol_select(signal.pair, True)

    tick = mt5.symbol_info_tick(signal.pair)
    order_type = mt5.ORDER_TYPE_BUY if signal.side == "buy" else mt5.ORDER_TYPE_SELL
    price = tick.ask if signal.side == "buy" else tick.bid
    volume = round(signal.units / 100_000, 2)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": signal.pair,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": signal.stop,
        "tp": signal.target,
        "deviation": int(cfg["execution"]["mt5_deviation_points"]),
        "magic": int(cfg["execution"]["mt5_magic"]),
        "comment": "atr_ema_rsi_scalper",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="ATR-ranked EMA/RSI forex scalper")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["scan", "backtest", "live-mt5"], default="scan")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "scan":
        signals = scan(cfg)
        if not signals:
            print("No valid signals right now.")
        for signal in signals:
            print(format_signal(signal))
    elif args.mode == "backtest":
        backtest(cfg)
    elif args.mode == "live-mt5":
        live_mt5(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
