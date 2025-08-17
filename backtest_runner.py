#!/usr/bin/env python3
"""
Research Mode backtester for StructureBot
- Fetches historical OHLCV (via CCXT; default binance)
- Replays BOS/SFP logic bar-by-bar
- Simulates entries/exits (limit at zone edge, SL/TP1)
- Writes results to Supabase (table: signals) with is_backtest=true
- Also saves CSVs under /backtests

Requirements (add to requirements.txt if missing):
  ccxt>=4.3.0
  pandas>=2.2.0
  numpy>=1.26.0
  pyarrow>=15.0.0
  PyYAML>=6.0
  requests>=2.32.0
"""

import os
import sys
import time
import json
import yaml
import math
import pathlib
import argparse
import datetime as dt
from typing import Optional, Dict, List, Tuple

import requests
import pandas as pd
import numpy as np

# ----- optional import; raise a friendly error if missing
try:
    import ccxt
except Exception as e:
    ccxt = None


# ==============================
# Utilities
# ==============================
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_ms(t: dt.datetime) -> int:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() * 1000)


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def parse_yaml_maybe(s: str):
    """Try to parse a YAML string; on failure return s."""
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def load_cfg() -> dict:
    """
    Load config from STRUCTURE_CONFIG (YAML string) if present,
    otherwise from config.yml. Always return a dict.
    """
    raw = os.environ.get("STRUCTURE_CONFIG")
    if raw:
        cfg = parse_yaml_maybe(raw)
        if isinstance(cfg, dict):
            return cfg
        # Allow case where env var is a path to a file
        if isinstance(cfg, str) and pathlib.Path(cfg).exists():
            with open(cfg, "r") as f:
                return yaml.safe_load(f)
        # As a last resort, treat it as empty dict
        return {}
    # Fallback to file
    if pathlib.Path("config.yml").exists():
        with open("config.yml", "r") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    return {}


# ==============================
# Supabase lightweight client
# ==============================
class Supa:
    def __init__(self, url: str, service_key: str, table: str):
        self.url = url.rstrip("/")
        self.key = service_key
        self.table = table
        self.s = requests.Session()
        self.s.headers.update({
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        })

    def insert_signals(self, rows: List[dict]) -> None:
        if not rows:
            return
        endpoint = f"{self.url}/rest/v1/{self.table}"
        # on_conflict dedupe on dedupe_key if your table has it
        params = {"on_conflict": "dedupe_key"}
        r = self.s.post(endpoint, params=params, data=json.dumps(rows))
        if r.status_code >= 300:
            print(f"[DB][ERR] insert {r.status_code}: {r.text}", file=sys.stderr)

    def patch_signal(self, dedupe_key: str, updates: dict) -> None:
        if not dedupe_key:
            return
        endpoint = f"{self.url}/rest/v1/{self.table}"
        params = {"dedupe_key": f"eq.{dedupe_key}", "select": "*"}
        r = self.s.patch(endpoint, params=params, data=json.dumps(updates))
        if r.status_code >= 300:
            print(f"[DB][ERR] patch {r.status_code}: {r.text}", file=sys.stderr)


# ==============================
# Candle fetching (CCXT + cache)
# ==============================
def get_exchange(name: str):
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Add it to requirements.txt.")
    name = (name or "binance").lower()
    if not hasattr(ccxt, name):
        raise RuntimeError(f"Unknown exchange for ccxt: {name}")
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    # Use futures where available for continuity
    if hasattr(ex, "options"):
        ex.options.setdefault("defaultType", "future")
    return ex


def fetch_ohlcv_cached(symbol: str, timeframe: str, since_ms: int, until_ms: int,
                       cache_dir: str, exchange_name: str = "binance") -> pd.DataFrame:
    ensure_dir(cache_dir)
    ex = get_exchange(exchange_name)
    tf_ms = ex.parse_timeframe(timeframe) * 1000

    cache_file = pathlib.Path(cache_dir) / f"{exchange_name}_{symbol.replace('/', '-')}_{timeframe}.parquet"
    cached = None
    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
        except Exception:
            cached = None

    rows: List[dict] = [] if cached is None else cached.to_dict("records")
    start_ms = since_ms if cached is None else int(cached["t"].iloc[-1]) + tf_ms

    if start_ms <= until_ms:
        fetch_from = start_ms
        while fetch_from < until_ms:
            limit = 1000
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_from, limit=limit)
            if not batch:
                break
            for o in batch:
                if o[0] > until_ms:
                    break
                rows.append({"t": int(o[0]), "o": float(o[1]), "h": float(o[2]),
                             "l": float(o[3]), "c": float(o[4]), "v": float(o[5])})
            fetch_from = batch[-1][0] + tf_ms
            time.sleep(ex.rateLimit / 1000.0)  # gentle

    df = pd.DataFrame(rows).drop_duplicates("t").sort_values("t")
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


# ==============================
# Strategy logic (BOS/SFP + zone)
# ==============================
def series_atr(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
    prev_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def pivots(h: pd.Series, l: pd.Series, left: int = 2, right: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Boolean series for swing highs/lows using a simple fixed window."""
    w = left + right + 1
    sh = (h.shift(left).rolling(w).max() == h) & (h == h.rolling(w, center=True).max())
    sl = (l.shift(left).rolling(w).min() == l) & (l == l.rolling(w, center=True).min())
    return sh.fillna(False), sl.fillna(False)


def market_regime(close: pd.Series, lookback: int = 50) -> str:
    if len(close) < lookback + 5:
        return "ranging"
    x = np.arange(lookback)
    y = close.iloc[-lookback:].values
    slope = np.polyfit(x, y, 1)[0]
    return "trending" if abs(slope) > (np.std(y) * 0.002) else "ranging"


def directional_bias(close: pd.Series, lookback: int = 50) -> str:
    if len(close) < lookback:
        return "neutral"
    return "bullish" if close.iloc[-1] > close.rolling(lookback).mean().iloc[-1] else "bearish"


def detect_signal(df: pd.DataFrame) -> Optional[Dict]:
    """
    Detect either BOS (body breaks last swing) or SFP (wick through, close back).
    Builds the wick->body zone from the impulse bar (previous bar).
    """
    if len(df) < 80:
        return None
    o, h, l, c = df.o, df.h, df.l, df.c
    sh, sl = pivots(h, l, 2, 2)

    look = df.iloc[-60:]
    last_swing_high = look[sh].h.max() if (look[sh].shape[0] > 0) else None
    last_swing_low  = look[sl].l.min() if (look[sl].shape[0] > 0) else None

    atrv = series_atr(h, l, c, 14).iloc[-1]
    reg = market_regime(c)
    b   = directional_bias(c)

    # SFP
    sfp_kind, sfp_level = None, None
    if last_swing_high is not None and h.iloc[-1] > last_swing_high and c.iloc[-1] < last_swing_high:
        sfp_kind, sfp_level = "bearish", float(last_swing_high)
    if last_swing_low is not None and l.iloc[-1] < last_swing_low and c.iloc[-1] > last_swing_low:
        if sfp_kind is None:
            sfp_kind, sfp_level = "bullish", float(last_swing_low)

    # BOS
    bos_kind, bos_level = None, None
    if last_swing_high is not None and c.iloc[-1] > last_swing_high:
        bos_kind, bos_level = "bullish", float(last_swing_high)
    if last_swing_low is not None and c.iloc[-1] < last_swing_low:
        if bos_kind is None:
            bos_kind, bos_level = "bearish", float(last_swing_low)

    # prefer BOS over SFP if both
    if bos_kind:
        direction = "bullish" if bos_kind == "bullish" else "bearish"
        imp = df.iloc[-2]  # impulse bar
        if direction == "bullish":
            z_top = float(max(imp.o, imp.c))
            z_bot = float(min(imp.l, imp.o, imp.c))
        else:
            z_top = float(max(imp.h, imp.o, imp.c))
            z_bot = float(min(imp.o, imp.c))
        entry = z_bot if direction == "bullish" else z_top
        stop  = entry - atrv*0.8 if direction == "bullish" else entry + atrv*0.8
        tp1   = entry + (entry - stop)*1.6 if direction == "bullish" else entry - (stop - entry)*1.6
        score = 80.0 + (10.0 if reg == "trending" else 0.0)
        if (direction == "bullish" and b == "bullish") or (direction == "bearish" and b == "bearish"):
            score += 10.0
        clean = abs(z_top - z_bot) / max(1e-8, atrv)
        score += max(0.0, 10.0 - min(10.0, clean*2))
        return {
            "type": "BOS", "direction": direction,
            "zone_kind": "bullish" if direction == "bullish" else "bearish",
            "zone_top": z_top, "zone_bottom": z_bot,
            "level": float(bos_level),
            "entry": float(entry), "stop": float(stop), "tp1": float(tp1),
            "atr": float(atrv), "score": round(score, 2),
            "reasons": f"bos•reg:{reg}•bias:{b}•zoneW:{abs(z_top-z_bot):.4f}•atr:{atrv:.4f}",
        }

    if sfp_kind:
        direction = "bearish" if sfp_kind == "bearish" else "bullish"
        imp = df.iloc[-2]
        if direction == "bullish":
            z_top = float(max(imp.o, imp.c))
            z_bot = float(min(imp.l, imp.o, imp.c))
        else:
            z_top = float(max(imp.h, imp.o, imp.c))
            z_bot = float(min(imp.o, imp.c))
        entry = z_bot if direction == "bullish" else z_top
        stop  = entry - 0.9*series_atr(df.h, df.l, df.c, 14).iloc[-1] if direction == "bullish" else entry + 0.9*series_atr(df.h, df.l, df.c, 14).iloc[-1]
        tp1   = entry + (entry - stop)*1.6 if direction == "bullish" else entry - (stop - entry)*1.6
        score = 70.0 + (10.0 if reg == "ranging" else 0.0)
        if (direction == "bullish" and b == "bullish") or (direction == "bearish" and b == "bearish"):
            score += 10.0
        clean = abs(z_top - z_bot) / max(1e-8, series_atr(df.h, df.l, df.c, 14).iloc[-1])
        score += max(0.0, 10.0 - min(10.0, clean*2))
        return {
            "type": "SFP", "direction": direction,
            "zone_kind": "bullish" if direction == "bullish" else "bearish",
            "zone_top": z_top, "zone_bottom": z_bot,
            "level": float(sfp_level),
            "entry": float(entry), "stop": float(stop), "tp1": float(tp1),
            "atr": float(series_atr(df.h, df.l, df.c, 14).iloc[-1]),
            "score": round(score, 2),
            "reasons": f"sfp•reg:{reg}•bias:{b}•zoneW:{abs(z_top-z_bot):.4f}",
        }

    return None


# ==============================
# Outcome simulation
# ==============================
def simulate_forward(df: pd.DataFrame, start_i: int, direction: str,
                     entry: float, stop: float, tp1: float) -> Tuple[str, float, int, Optional[int], Optional[int], Optional[float]]:
    """
    Walk future bars to determine outcome:
      - wait for limit entry hit
      - after entry, SL has priority if both hit in same bar
      - returns: outcome, rr, minutes_in_trade, enter_t, exit_t, exit_price
    """
    enter_t = None
    exit_t, exit_px = None, None
    rr = abs((tp1 - entry) / (entry - stop)) if (entry - stop) != 0 else 0.0

    def touched(px_low, px_high, price) -> bool:
        return px_low <= price <= px_high

    for i in range(start_i + 1, len(df)):
        row = df.iloc[i]
        low, high, t = row["l"], row["h"], int(row["t"])

        if enter_t is None:
            # limit fill
            if touched(low, high, entry):
                enter_t = t
                # after fill, evaluate SL first (adverse priority)
                if direction == "bullish":
                    if low <= stop:
                        return "sl", -1.0, 0, enter_t, t, stop
                    if high >= tp1:
                        return "tp", rr, 0, enter_t, t, tp1
                else:
                    if high >= stop:
                        return "sl", -1.0, 0, enter_t, t, stop
                    if low <= tp1:
                        return "tp", rr, 0, enter_t, t, tp1
        else:
            if direction == "bullish":
                if low <= stop:
                    return "sl", -1.0, int((t - enter_t) / 60000), enter_t, t, stop
                if high >= tp1:
                    return "tp", rr, int((t - enter_t) / 60000), enter_t, t, tp1
            else:
                if high >= stop:
                    return "sl", -1.0, int((t - enter_t) / 60000), enter_t, t, stop
                if low <= tp1:
                    return "tp", rr, int((t - enter_t) / 60000), enter_t, t, tp1

    if enter_t is None:
        return "missed", 0.0, 0, None, None, None
    return "open", 0.0, int((int(df.iloc[-1]["t"]) - enter_t) / 60000), enter_t, None, None


# ==============================
# Backtest runner
# ==============================
def run_backtest(cfg: dict, args):
    # Robust: cfg might be string if something upstream passed it wrong
    if isinstance(cfg, str):
        cfg = parse_yaml_maybe(cfg)
        if not isinstance(cfg, dict):
            cfg = {}

    research = cfg.get("research", {})
    if not isinstance(research, dict):
        research = {}

    exchange = args.exchange or research.get("exchange", "binance")
    symbols = args.symbols or research.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframes = args.timeframes or research.get("timeframes", ["1m"])
    days = int(args.days or research.get("days", 30))
    min_score = float(args.min_score or research.get("min_score", 0))
    cache_dir = args.cache or research.get("cache_dir", "cache")
    save_csv = True if args.save_csv is None else args.save_csv
    push_db = True if args.push_db is None else args.push_db
    signals_table = research.get("signals_table", cfg.get("supabase", {}).get("signals_table", "signals"))

    sb_url = os.environ.get("SUPABASE_URL")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    supa = None
    if push_db and sb_url and sb_key:
        supa = Supa(sb_url, sb_key, signals_table)
        print(f"[DB] Connected {sb_url} / table={signals_table}")
    elif push_db:
        print("[DB] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY; will skip DB writes.")

    until = utc_now()
    since = until - dt.timedelta(days=days)
    since_ms, until_ms = to_ms(since), to_ms(until)

    print(f"[RESEARCH] ex={exchange} symbols={symbols} tfs={timeframes} range={since.isoformat()} → {until.isoformat()} min_score={min_score}")

    for sym in symbols:
        for tf in timeframes:
            print(f"[RUN] {sym} {tf} … fetch/calc")
            df = fetch_ohlcv_cached(sym, tf, since_ms, until_ms, cache_dir, exchange)
            if df.empty:
                print(f"[WARN] No data for {sym} {tf}")
                continue

            out_rows: List[dict] = []
            window = max(300, 80)
            for i in range(window, len(df) - 1):
                sub = df.iloc[i - window:i + 1].copy()
                sig = detect_signal(sub)
                if not sig:
                    continue
                if sig["score"] < min_score:
                    continue

                # simulate forward from i
                outcome, rr, mins, enter_t, exit_t, exit_px = simulate_forward(
                    df, i, sig["direction"], sig["entry"], sig["stop"], sig["tp1"]
                )

                created_ts = dt.datetime.fromtimestamp(df.iloc[i]["t"] / 1000.0, tz=dt.timezone.utc)
                row = {
                    "created_at": created_ts.isoformat(),
                    "symbol": sym, "timeframe": tf,
                    "type": sig["type"], "direction": sig["direction"],
                    "zone_kind": sig["zone_kind"],
                    "zone_top": float(sig["zone_top"]), "zone_bottom": float(sig["zone_bottom"]),
                    "level": float(sig["level"]), "idx": i,
                    "entry": float(sig["entry"]), "stop": float(sig["stop"]), "tp1": float(sig["tp1"]),
                    "atr": float(sig["atr"]), "score": float(sig["score"]),
                    "reasons": sig["reasons"], "is_backtest": True,
                    "dedupe_key": f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],6)}:{int(df.iloc[i]['t'])}",
                    "entered_at": None if enter_t is None else dt.datetime.fromtimestamp(enter_t / 1000.0, tz=dt.timezone.utc).isoformat(),
                    "exited_at": None if exit_t is None else dt.datetime.fromtimestamp(exit_t / 1000.0, tz=dt.timezone.utc).isoformat(),
                    "exit_price": exit_px,
                    "rr_achieved": rr,
                    "outcome": outcome,
                    "minutes_in_trade": mins
                }
                out_rows.append(row)

                # periodic flush
                if supa and len(out_rows) >= 250:
                    supa.insert_signals(out_rows)
                    out_rows = []

            # final flush
            if supa and out_rows:
                supa.insert_signals(out_rows)

            if save_csv:
                outdir = pathlib.Path("backtests")
                ensure_dir(outdir.as_posix())
                outfile = outdir / f"{sym.replace('/','-')}_{tf}_{days}d.csv"
                pd.DataFrame(out_rows).to_csv(outfile, index=False)
                print(f"[SAVE] {outfile} ({len(out_rows)} rows)")

    print("[DONE] research run complete.")


# ==============================
# CLI
# ==============================
def build_argparser():
    p = argparse.ArgumentParser(description="StructureBot Research Mode Backtester")
    p.add_argument("--exchange", type=str, default=None, help="ccxt exchange (default from config, else binance)")
    p.add_argument("--symbols", type=lambda s: [x.strip() for x in s.split(",")], default=None,
                   help="comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    p.add_argument("--timeframes", type=lambda s: [x.strip() for x in s.split(",")], default=None,
                   help="comma-separated tfs, e.g. 1m,5m,15m")
    p.add_argument("--days", type=int, default=None, help="lookback days")
    p.add_argument("--min_score", type=float, default=None, help="minimum score filter")
    p.add_argument("--cache", type=str, default=None, help="cache directory")
    p.add_argument("--no-db", dest="push_db", action="store_false", help="disable Supabase writes")
    p.add_argument("--no-csv", dest="save_csv", action="store_false", help="disable CSV saving")
    return p


if __name__ == "__main__":
    cfg = load_cfg()
    args = build_argparser().parse_args()
    try:
        run_backtest(cfg, args)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        raise
