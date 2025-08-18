#!/usr/bin/env python3
"""
StructureBot — Research Mode Backtester
- Fetches historical OHLCV via CCXT (default: bybit, futures if available)
- Replays BOS & SFP logic bar-by-bar
- Simulates limit entry at zone edge, SL priority, TP1 (~1.6R)
- Upserts results to Supabase (tagged is_backtest=true)
- Saves CSVs under /backtests/

Reqs:
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

# ---- optional import guard
try:
    import ccxt
except Exception:
    ccxt = None


# =============== small utils ===============
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def to_ms(t: dt.datetime) -> int:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() * 1000)

def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def parse_yaml_maybe(s: str):
    try:
        return yaml.safe_load(s)
    except Exception:
        return s

def load_cfg() -> dict:
    raw = os.environ.get("STRUCTURE_CONFIG")
    if raw:
        cfg = parse_yaml_maybe(raw)
        if isinstance(cfg, dict):
            return cfg
        if isinstance(cfg, str) and pathlib.Path(cfg).exists():
            with open(raw, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    if pathlib.Path("config.yml").exists():
        with open("config.yml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")


# =============== Supabase lite client ===============
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

    def insert(self, rows: List[dict]) -> None:
        if not rows:
            return
        endpoint = f"{self.url}/rest/v1/{self.table}"
        params = {"on_conflict": "dedupe_key"}
        r = self.s.post(endpoint, params=params, data=json.dumps(rows))
        if r.status_code >= 300:
            print(f"[DB][ERR] insert {r.status_code}: {r.text}", file=sys.stderr)


# =============== CCXT + helpers ===============
def get_exchange(name: str):
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Add to requirements.txt")
    name = (name or "bybit").lower()
    if not hasattr(ccxt, name):
        raise RuntimeError(f"Unknown ccxt exchange: {name}")
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    # prefer futures if supported
    if hasattr(ex, "options"):
        ex.options.setdefault("defaultType", "future")
    return ex

def resolve_symbol(ex, sym: str) -> str:
    """Handle Bybit quirk: BTC/USDT vs BTC/USDT:USDT."""
    try:
        mkts = ex.load_markets()
        if sym in mkts:
            return sym
        if sym.endswith("/USDT") and f"{sym}:USDT" in mkts:
            return f"{sym}:USDT"
        if sym.endswith(":USDT") and sym.replace(":USDT", "") in mkts:
            return sym.replace(":USDT", "")
    except Exception:
        pass
    return sym

def fetch_ohlcv_cached(symbol: str, timeframe: str, since_ms: int, until_ms: int,
                       cache_dir: str, exchange_name: str) -> pd.DataFrame:
    ensure_dir(cache_dir)
    ex = get_exchange(exchange_name)
    symbol = resolve_symbol(ex, symbol)
    tf_ms = ex.parse_timeframe(timeframe) * 1000

    safe_sym = symbol.replace("/", "-").replace(":", "-")
    cache_file = pathlib.Path(cache_dir) / f"{exchange_name}_{safe_sym}_{timeframe}.parquet"
    cached = None
    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
        except Exception:
            cached = None

    rows = [] if cached is None else cached.to_dict("records")
    start_ms = since_ms if cached is None else int(cached["t"].iloc[-1]) + tf_ms

    if start_ms <= until_ms:
        fetch_from = start_ms
        while fetch_from < until_ms:
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_from, limit=1000)
            except Exception as e:
                print(f"[FETCH][ERR] {symbol} {timeframe}: {e}")
                break
            if not batch:
                break
            for o in batch:
                if o[0] > until_ms:
                    break
                rows.append({"t": int(o[0]), "o": float(o[1]), "h": float(o[2]),
                             "l": float(o[3]), "c": float(o[4]), "v": float(o[5])})
            fetch_from = batch[-1][0] + tf_ms
            time.sleep(getattr(ex, "rateLimit", 250) / 1000.0)

    df = pd.DataFrame(rows).drop_duplicates("t").sort_values("t")
    if not df.empty:
        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"[CACHE][WARN] write parquet failed: {e}")
    return df


# =============== Strategy core ===============
def series_atr(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
    prev_close = c.shift(1)
    tr = pd.concat([(h - l).abs(),
                    (h - prev_close).abs(),
                    (l - prev_close).abs()], axis=1).max(axis=1)
    # simple mean is fine for backtests (EMA/RMA not required)
    return tr.rolling(length).mean()

def pivots(h: pd.Series, l: pd.Series, left: int = 2, right: int = 2) -> Tuple[pd.Series, pd.Series]:
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
    Detect BOS / SFP using the last ~60 bars window.
    Compute swing masks on the *window slice*, then use .loc to avoid reindex warnings.
    """
    if len(df) < 80:
        return None

    look = df.iloc[-60:].copy()
    sh_mask, sl_mask = pivots(look["h"], look["l"], 2, 2)

    last_swing_high = float(look.loc[sh_mask, "h"].max()) if sh_mask.any() else None
    last_swing_low  = float(look.loc[sl_mask, "l"].min()) if sl_mask.any() else None

    atrv = float(series_atr(df["h"], df["l"], df["c"], 14).iloc[-1])
    reg  = market_regime(df["c"])
    bias = directional_bias(df["c"])

    h, l, c = float(df["h"].iloc[-1]), float(df["l"].iloc[-1]), float(df["c"].iloc[-1])

    # --- SFP: wick through, close back in
    sfp_kind, sfp_level = None, None
    if last_swing_high is not None and h > last_swing_high and c < last_swing_high:
        sfp_kind, sfp_level = "bearish", last_swing_high
    if last_swing_low is not None and l < last_swing_low and c > last_swing_low:
        if sfp_kind is None:
            sfp_kind, sfp_level = "bullish", last_swing_low

    # --- BOS: body close breaks swing
    bos_kind, bos_level = None, None
    if last_swing_high is not None and c > last_swing_high:
        bos_kind, bos_level = "bullish", last_swing_high
    if last_swing_low is not None and c < last_swing_low:
        if bos_kind is None:
            bos_kind, bos_level = "bearish", last_swing_low

    # Zone from the previous (impulse) bar
    imp = df.iloc[-2]

    def build_zone(direction: str) -> Tuple[float, float, float, float, float]:
        """
        Returns (z_top, z_bot, entry, stop, tp1).
        - bullish: use wick->body under impulse (entry near lower edge)
        - bearish: use body->wick above impulse (entry near upper edge)
        """
        if direction == "bullish":
            z_top = float(max(imp.o, imp.c))                 # body high
            z_bot = float(min(imp.l, imp.o, imp.c))          # wick or body low
            entry = z_bot
            stop  = entry - 0.8 * atrv
            tp1   = entry + (entry - stop) * 1.6
        else:
            z_top = float(max(imp.h, imp.o, imp.c))          # wick or body high
            z_bot = float(min(imp.o, imp.c))                 # body low
            entry = z_top
            stop  = entry + 0.8 * atrv
            tp1   = entry - (stop - entry) * 1.6
        return z_top, z_bot, entry, stop, tp1

    def score_base(direction: str, reg: str, bias: str, width: float, atrv: float, base: float) -> float:
        s = base
        if (direction == "bullish" and bias == "bullish") or (direction == "bearish" and bias == "bearish"):
            s += 10.0
        if reg == ("trending" if base >= 80 else "ranging"):
            s += 10.0
        clean_penalty = max(0.0, min(10.0, (width / max(1e-9, atrv)) * 2))
        return round(s + max(0.0, 10.0 - clean_penalty), 2)

    # Prefer BOS if both fire (trend continuation > fade)
    if bos_kind:
        direction = "bullish" if bos_kind == "bullish" else "bearish"
        z_top, z_bot, entry, stop, tp1 = build_zone(direction)
        width = abs(z_top - z_bot)
        score = score_base(direction, reg, bias, width, atrv, base=80.0)
        return {
            "type": "BOS", "direction": direction,
            "zone_kind": "bullish" if direction == "bullish" else "bearish",
            "zone_top": z_top, "zone_bottom": z_bot,
            "level": float(bos_level),
            "entry": float(entry), "stop": float(stop), "tp1": float(tp1),
            "atr": float(atrv), "score": score,
            "reasons": f"bos•reg:{reg}•bias:{bias}•zoneW:{width:.6f}•atr:{atrv:.6f}",
        }

    if sfp_kind:
        direction = "bearish" if sfp_kind == "bearish" else "bullish"
        z_top, z_bot, entry, stop, tp1 = build_zone(direction)
        width = abs(z_top - z_bot)
        score = score_base(direction, reg, bias, width, atrv, base=70.0)
        return {
            "type": "SFP", "direction": direction,
            "zone_kind": "bullish" if direction == "bullish" else "bearish",
            "zone_top": z_top, "zone_bottom": z_bot,
            "level": float(sfp_level),
            "entry": float(entry), "stop": float(stop), "tp1": float(tp1),
            "atr": float(atrv), "score": score,
            "reasons": f"sfp•reg:{reg}•bias:{bias}•zoneW:{width:.6f}•atr:{atrv:.6f}",
        }

    return None


# =============== Outcome simulation ===============
def simulate_forward(df: pd.DataFrame, start_i: int, direction: str,
                     entry: float, stop: float, tp1: float) -> Tuple[str, float, int, Optional[int], Optional[int], Optional[float]]:
    """
    Walk forward:
      - wait for limit entry
      - after entry, SL has priority if both hit same bar
    Returns: (outcome, rr, minutes_in_trade, enter_t, exit_t, exit_price)
    """
    enter_t = None
    exit_t, exit_px = None, None
    rr = abs((tp1 - entry) / (entry - stop)) if (entry - stop) != 0 else 0.0

    def touched(low, high, price) -> bool:
        return low <= price <= high

    for i in range(start_i + 1, len(df)):
        row = df.iloc[i]
        low, high, t = float(row["l"]), float(row["h"]), int(row["t"])

        if enter_t is None:
            if touched(low, high, entry):
                enter_t = t
                # same-bar resolution after fill: SL first
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


# =============== Runner ===============
def run_backtest(cfg: dict, args):
    if isinstance(cfg, str):
        cfg = parse_yaml_maybe(cfg) or {}
        if not isinstance(cfg, dict):
            cfg = {}

    research = cfg.get("research", {})
    if not isinstance(research, dict):
        research = {}

    exchange   = args.exchange   or research.get("exchange", "bybit")
    symbols    = args.symbols    or research.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframes = args.timeframes or research.get("timeframes", ["1m"])
    days       = int(args.days   or research.get("days", 30))
    min_score  = float(args.min_score or research.get("min_score", 70))
    cache_dir  = args.cache      or research.get("cache_dir", "cache")
    save_csv   = True if args.save_csv is None else args.save_csv
    push_db    = True if args.push_db  is None else args.push_db
    signals_table = research.get("signals_table", cfg.get("supabase", {}).get("signals_table", "signals"))
    debug = env_flag("RESEARCH_DEBUG", False)

    sb_url = os.environ.get("SUPABASE_URL")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    supa = None
    if push_db and sb_url and sb_key:
        supa = Supa(sb_url, sb_key, signals_table)
        print(f"[DB] Connected {sb_url} / table={signals_table}")
    elif push_db:
        print("[DB] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY — skipping DB writes")

    until = utc_now()
    since = until - dt.timedelta(days=days)
    since_ms, until_ms = to_ms(since), to_ms(until)

    print(f"[RESEARCH] exchange={exchange} symbols={symbols} tfs={timeframes} "
          f"range={since.isoformat()} → {until.isoformat()} min_score={min_score}")

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
                if not sig or sig["score"] < min_score:
                    continue

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
                    # include bar timestamp to dedupe per-detection
                    "dedupe_key": f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],6)}:{int(df.iloc[i]['t'])}",
                    "entered_at": None if enter_t is None else dt.datetime.fromtimestamp(enter_t / 1000.0, tz=dt.timezone.utc).isoformat(),
                    "exited_at": None if exit_t is None else dt.datetime.fromtimestamp(exit_t / 1000.0, tz=dt.timezone.utc).isoformat(),
                    "exit_price": float(exit_px) if exit_px is not None else None,
                    "rr_achieved": float(rr),
                    "outcome": outcome,
                    "minutes_in_trade": int(mins),
                }
                out_rows.append(row)

                if debug:
                    print(f"[SIG] {sym} {tf} {row['type']} {row['direction']} "
                          f"score={row['score']} outcome={outcome} rr={rr:.2f}")

                # periodic flush to DB
                if supa and len(out_rows) >= 250:
                    supa.insert(out_rows)
                    out_rows = []

            # final flush
            if supa and out_rows:
                supa.insert(out_rows)

            if save_csv:
                try:
                    ensure_dir("backtests")
                    outfile = pathlib.Path("backtests") / f"{sym.replace('/','-')}_{tf}_{days}d.csv"
                    pd.DataFrame(out_rows).to_csv(outfile, index=False)
                    print(f"[SAVE] {outfile} ({len(out_rows)} rows)")
                except Exception as e:
                    print(f"[SAVE][WARN] csv failed: {e}")

    print("[DONE] research run complete.")


# =============== CLI ===============
def build_argparser():
    p = argparse.ArgumentParser(description="StructureBot Research Mode Backtester")
    p.add_argument("--exchange", type=str, default=None)
    p.add_argument("--symbols", type=lambda s: [x.strip() for x in s.split(",")], default=None)
    p.add_argument("--timeframes", type=lambda s: [x.strip() for x in s.split(",")], default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--min_score", type=float, default=None)
    p.add_argument("--cache", type=str, default=None)
    p.add_argument("--no-db", dest="push_db", action="store_false")
    p.add_argument("--no-csv", dest="save_csv", action="store_false")
    return p


if __name__ == "__main__":
    cfg = load_cfg()
    args = build_argparser().parse_args()
    try:
        run_backtest(cfg, args)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        raise
