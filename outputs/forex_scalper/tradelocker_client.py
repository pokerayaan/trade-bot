from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests


@dataclass
class TradeLockerCredentials:
    email: str
    password: str
    server: str
    env: str
    account_id: str = ""
    account_number: str = ""


class TradeLockerClient:
    def __init__(self, cfg: dict[str, Any], credentials: TradeLockerCredentials):
        self.cfg = cfg
        self.credentials = credentials
        self.env = credentials.env.rstrip("/")
        self.api_base = cfg["tradelocker"].get("api_base", f"{self.env}/backend-api").rstrip("/")
        self.session = requests.Session()
        self.access_token = ""
        self.account_id = credentials.account_id or cfg["tradelocker"].get("account_id", "")
        self.account_number = credentials.account_number or cfg["tradelocker"].get("account_number", "")
        self.account: dict[str, Any] = {}
        self.route_id: str = ""
        self._instruments_cache: list[dict[str, Any]] | None = None

    @classmethod
    def from_env(cls, cfg: dict[str, Any]) -> "TradeLockerClient":
        tl_cfg = cfg["tradelocker"]
        creds = TradeLockerCredentials(
            email=os.getenv("TRADELOCKER_EMAIL", ""),
            password=os.getenv("TRADELOCKER_PASSWORD", ""),
            server=os.getenv("TRADELOCKER_SERVER", tl_cfg.get("server", "")),
            env=os.getenv("TRADELOCKER_ENV", tl_cfg.get("env", "https://demo.tradelocker.com")),
            account_id=os.getenv("TRADELOCKER_ACCOUNT_ID", tl_cfg.get("account_id", "")),
            account_number=os.getenv("TRADELOCKER_ACCOUNT_NUMBER", tl_cfg.get("account_number", "")),
        )
        missing = [name for name, value in vars(creds).items() if name in {"email", "password", "server", "env"} and not value]
        if missing:
            raise ValueError(f"Missing TradeLocker credentials: {', '.join(missing)}")
        return cls(cfg, creds)

    def login(self) -> None:
        payloads = [
            {"email": self.credentials.email, "password": self.credentials.password, "server": self.credentials.server},
            {"username": self.credentials.email, "password": self.credentials.password, "server": self.credentials.server},
        ]
        response = self._first_success("post", "auth", payloads=payloads)
        data = response.json()
        self.access_token = (
            data.get("accessToken")
            or data.get("access_token")
            or data.get("token")
            or data.get("jwt")
            or data.get("data", {}).get("accessToken")
            or data.get("data", {}).get("token")
        )
        if not self.access_token:
            raise RuntimeError(f"TradeLocker login succeeded but no token was found in response keys: {list(data.keys())}")
        self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})
        self.discover_account()

    def discover_account(self) -> None:
        response = self._first_success("get", "accounts")
        data = response.json()
        accounts = self._extract_list(data)
        if not accounts:
            raise RuntimeError("Logged in, but no TradeLocker accounts were returned.")

        # If account_id was pre-configured, find the matching account; else use first
        if self.account_id:
            matched = next(
                (
                    a for a in accounts
                    if str(a.get("id") or a.get("accountId") or a.get("tradableInstrumentAccountId") or "") == str(self.account_id)
                ),
                None,
            )
            self.account = matched or accounts[0]
        else:
            self.account = accounts[0]
            self.account_id = str(
                self.account.get("id")
                or self.account.get("accountId")
                or self.account.get("tradableInstrumentAccountId")
                or ""
            )
            if not self.account_id:
                raise RuntimeError(f"Could not infer account_id from account response: {self.account}")

        self.account_number = self.account_number or str(
            self.account.get("accNum") or self.account.get("accountNumber") or ""
        )
        # Capture routeId so place_market_order can always include it
        self.route_id = str(
            self.account.get("routeId")
            or self.account.get("tradeRouteId")
            or (self.account.get("tradeRoute") or {}).get("id", "")
            or ""
        )
        if not self.route_id:
            import json as _json
            print(f"[DEBUG] route_id not found. Account keys: {list(self.account.keys())}")
            print(f"[DEBUG] Full account: {_json.dumps(self.account, default=str)[:600]}")
        else:
            print(f"[DEBUG] route_id resolved: {self.route_id}")
        self.session.headers.update({"accountId": self.account_id})
        if self.account_number:
            self.session.headers.update({"accNum": self.account_number})

    def instruments(self) -> list[dict[str, Any]]:
        if self._instruments_cache is not None:
            return self._instruments_cache
        response = self._first_success("get", "instruments", path_params={"account_id": self.account_id})
        self._instruments_cache = self._extract_list(response.json())
        return self._instruments_cache

    def debug_accounts_response(self) -> Any:
        response = self._first_success("get", "accounts")
        return response.json()

    def debug_instruments_response(self) -> Any:
        response = self._first_success("get", "instruments", path_params={"account_id": self.account_id})
        return response.json()

    def resolve_instrument(self, pair: str) -> dict[str, Any] | None:
        override = self.cfg["tradelocker"].get("symbol_overrides", {}).get(pair)
        for instrument in self.instruments():
            symbol = self._instrument_symbol(instrument)
            instrument_id = str(instrument.get("id") or instrument.get("tradableInstrumentId") or instrument.get("instrumentId") or "")
            if override and instrument_id == str(override):
                return instrument
            normalized = self._normalize_symbol(symbol)
            if normalized == pair.upper() or normalized.startswith(pair.upper()):
                return instrument
        return None

    def candles(self, pair: str, bars: int, resolution: str = "5m") -> pd.DataFrame:
        instrument = self.resolve_instrument(pair)
        if not instrument:
            raise RuntimeError(f"Could not resolve TradeLocker instrument for {pair}")
        instrument_id = str(instrument.get("id") or instrument.get("tradableInstrumentId") or instrument.get("instrumentId"))
        now = datetime.now(timezone.utc)
        minutes = self._resolution_minutes(resolution)
        start = now - timedelta(minutes=minutes * bars * 2)
        tl_resolution = self._tradelocker_resolution(resolution)
        params_candidates = [
            {
                "tradableInstrumentId": instrument_id,
                "resolution": tl_resolution,
                "from": int(start.timestamp() * 1000),
                "to": int(now.timestamp() * 1000),
                "countBack": bars,
            },
            {
                "tradableInstrumentId": instrument_id,
                "resolution": tl_resolution,
                "from": int(start.timestamp()),
                "to": int(now.timestamp()),
                "countBack": bars,
            },
            {
                "tradableInstrumentId": instrument_id,
                "resolution": resolution,
                "startTime": start.isoformat(),
                "endTime": now.isoformat(),
                "limit": bars,
            },
            {
                "instrumentId": instrument_id,
                "resolution": resolution,
                "timeframe": resolution,
                "limit": bars,
                "bars": bars,
            },
        ]
        response = None
        errors = []
        for params in params_candidates:
            try:
                response = self._first_success(
                    "get",
                    "candles",
                    path_params={"account_id": self.account_id, "instrument_id": instrument_id},
                    params=params,
                )
                break
            except RuntimeError as exc:
                errors.append(str(exc))
        if response is None:
            raise RuntimeError("\n".join(errors[-2:]))
        rows = self._extract_list(response.json())
        if not rows:
            raise RuntimeError(f"No candles returned for {pair}. Raw response: {str(response.json())[:500]}")
        frame = pd.DataFrame(rows)
        rename = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "t": "time",
            "timestamp": "time",
        }
        frame = frame.rename(columns=rename)
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "time" in frame:
            frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
            frame = frame.set_index("time")
        else:
            frame.index = pd.date_range(end=datetime.now(timezone.utc), periods=len(frame), freq="5min")
        return frame[["open", "high", "low", "close"]].dropna()

    @staticmethod
    def _resolution_minutes(resolution: str) -> int:
        value = resolution.lower().strip()
        if value.endswith("m"):
            return int(value[:-1])
        if value.endswith("h"):
            return int(value[:-1]) * 60
        return 5

    @staticmethod
    def _tradelocker_resolution(resolution: str) -> str:
        value = resolution.lower().strip()
        if value.endswith("m"):
            return f"M{value[:-1]}"
        if value.endswith("h"):
            return f"H{value[:-1]}"
        return resolution.upper()

    def place_market_order(
        self,
        pair: str,
        side: str,
        units: int | float,
        stop: float | None = None,
        target: float | None = None,
    ) -> dict[str, Any]:
        instrument = self.resolve_instrument(pair)
        if not instrument:
            raise RuntimeError(f"Could not resolve TradeLocker instrument for {pair}")
        instrument_id = str(instrument.get("id") or instrument.get("tradableInstrumentId") or instrument.get("instrumentId"))
        side_upper = side.upper()
        side_lower = side.lower()
        base_payloads = [
            {
                "tradableInstrumentId": instrument_id,
                "routeId": self.route_id or None,
                "side": side_upper,
                "type": "MARKET",
                "qty": units,
            },
            {
                "tradableInstrumentId": instrument_id,
                "side": side_upper,
                "orderType": "MARKET",
                "quantity": units,
            },
            {
                "instrumentId": instrument_id,
                "side": side_lower,
                "type": "market",
                "volume": units,
            },
        ]
        payloads = []
        for payload in base_payloads:
            clean_payload = {key: value for key, value in payload.items() if value not in (None, "")}
            if stop is not None:
                clean_payload["stopLoss"] = stop
            if target is not None:
                clean_payload["takeProfit"] = target
            payloads.append(clean_payload)
        response = self._first_success(
            "post",
            "orders",
            path_params={"account_id": self.account_id},
            payloads=payloads,
        )
        return response.json()

    def _first_success(
        self,
        method: str,
        endpoint_group: str,
        payloads: list[dict[str, Any]] | None = None,
        path_params: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        endpoints = self.cfg["tradelocker"]["endpoints"][endpoint_group]
        payloads = payloads or [{}]
        errors = []
        for endpoint in endpoints:
            path = endpoint.format(**(path_params or {}))
            url = f"{self.api_base}{path}"
            for payload in payloads:
                try:
                    response = self.session.request(method, url, json=payload if method == "post" else None, params=params, timeout=20)
                    if response.status_code == 429:
                        time.sleep(2)
                        response = self.session.request(
                            method,
                            url,
                            json=payload if method == "post" else None,
                            params=params,
                            timeout=20,
                        )
                    if 200 <= response.status_code < 300:
                        return response
                    errors.append(f"{method.upper()} {path}: {response.status_code} {response.text[:180]}")
                except requests.RequestException as exc:
                    errors.append(f"{method.upper()} {path}: {exc}")
        raise RuntimeError("No TradeLocker endpoint candidate succeeded:\n" + "\n".join(errors))

    @staticmethod
    def _instrument_symbol(instrument: dict[str, Any]) -> str:
        candidates = [
            instrument.get("symbol"),
            instrument.get("name"),
            instrument.get("displayName"),
            instrument.get("tradableInstrument"),
            instrument.get("tradableInstrumentSymbol"),
            instrument.get("localizedName"),
            instrument.get("description"),
        ]
        tradable = instrument.get("tradableInstrument")
        if isinstance(tradable, dict):
            candidates.extend(
                [
                    tradable.get("symbol"),
                    tradable.get("name"),
                    tradable.get("displayName"),
                    tradable.get("localizedName"),
                ]
            )
        return next((str(value) for value in candidates if value), "")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return "".join(ch for ch in symbol.upper() if ch.isalnum())

    @staticmethod
    def _extract_list(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in (
                "data",
                "accounts",
                "instruments",
                "tradableInstruments",
                "tradableInstrument",
                "items",
                "result",
                "rows",
                "candles",
                "bars",
                "s",
                "d",
            ):
                value = data.get(key)
                if isinstance(value, list):
                    return value
                if isinstance(value, dict):
                    nested = TradeLockerClient._extract_list(value)
                    if nested:
                        return nested
            for value in data.values():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    return value
                if isinstance(value, dict):
                    nested = TradeLockerClient._extract_list(value)
                    if nested:
                        return nested
        return []
