#!/usr/bin/env python3
"""
StructureBot v2 — Research Backtester

- Fetches historical OHLCV (via CCXT, default binance futures if available)
- Detects BOS/SFP exactly like the live bot
- HARD FILTERS:
    * BOS only in trending regime, SFP only in ranging regime
    * Min ATR percentile (volatility floor)
    * Per-symbol/timeframe allow-list for types and hours (UTC)
- Builds plan with dynamic zone retest offset + ATR-padded stop
- Simulates forward:
    * Waits for limit entry
    * SL priority on bars that hit both SL and TP
- Upserts rows to Supabase (table configurable), tagged is_backtest=true
- Also saves CSVs per (symbol, timeframe) under ./backtests/

Env (Railway):
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  STRUCTURE_CONFIG=./config.yml (or inline YAML string)
  (optional) SUPABASE_TABLE=signals

Requires:
  ccxt>=4.3.0, numpy>=1.26, pandas>=2.2, pyarrow>=15, PyYAML>=6, requests>=2.32
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import pathlib
import argparse
import datetime as dt
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

# ---------- config utils ----------

def _parse_yaml_maybe(s: str):
    try:
        return yaml.safe_load(s)
    except Exception:
        return s

def load_cfg() -> dict:
    raw = os.environ.get("STRUCTURE_CONFIG")
    if raw:
        cfg = _parse_yaml_maybe(raw)
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

def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# ---------- supabase (light) ----------

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
            "Prefer": "resolution=merge-duplicates,return=representation",
            "Accept": "application/json",
        })

    def insert(self, rows: List[dict]) -> None:
        if not rows:
            return
        endpoint = f"{self.url}/rest/v1/{self.table}"
        params = {"on_conflict": "dedupe_key"}
        r = self.s.post(endpoint, params=params, data=json.dumps(rows))
        if r.status_code >= 300:
            print(f"[DB][ERR] insert {r.status_code}: {r.text}", file=sys.stderr)

# ---------- exchange + cache ----------

try:
    import ccxt
except Exception:
    ccxt = None

def get_exchange(name: str):
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    name = (name or "binance").lower()
    if not hasattr(ccxt, name):
        raise RuntimeError(f"Unknown exchange: {name}")
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    if hasattr(ex, "options"):
        ex.options.setdefault("defaultType", "future")
    return ex

def fetch_ohlcv_cached(symbol: str, timeframe: str, since_ms: int, until_ms: int,
                       cache_dir: str, exchange_name: str) -> pd.DataFrame:
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

    rows = [] if cached is None else cached.to_dict("records")
    start_ms = since_ms if cached is None else int(cached["t"].iloc[-1]) + tf_ms

    if start_ms <= until_ms:
        fetch_from = start_ms
        while fetch_from < until_ms:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_from, limit=1000)
            if not batch:
                break
            for o in batch:
                if o[0] > until_ms:
                    break
                rows.append({"t": int(o[0]), "o": float(o[1]), "h": float(o[2]),
                             "l": float(o[3]), "c": float(o[4]), "v": float(o[5])})
            fetch_from = batch[-1][0] + tf_ms
            time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(rows).drop_duplicates("t").sort_values("t")
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df

# ---------- analytics core (same logic as live) ----------

def atr_series(arr: np.ndarray, length: int = 14) -> np.ndarray:
    h, l, c = arr[:, 2], arr[:, 3], arr[:, 4]
    prev_close = np.roll(c, 1); prev_close[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    out = np.zeros_like(tr)
    alpha = 1.0 / length
    rma = tr[0]
    for i in range(len(tr)):
        rma = alpha * tr[i] + (1 - alpha) * rma
        out[i] = rma
    return out

def atr_last(arr: np.ndarray, length: int = 14) -> float:
    return float(atr_series(arr, length)[-1])

def percentile_of_last(values: np.ndarray, lookback: int = 200) -> float:
    if len(values) < lookback:
        lookback = len(values)
    window = values[-lookback:]
    last = window[-1]
    rank = (window <= last).mean() * 100.0
    return float(rank)

def candle_body_ratio(o: float, c: float, h: float, l: float) -> float:
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    return float(body / rng)

class Zone:
    __slots__ = ("kind","top","bottom","impulse_end_idx","strength")
    def __init__(self, kind: str, top: float, bottom: float, i_end: int, strength: float):
        self.kind = kind
        self.top = float(top)
        self.bottom = float(bottom)
        self.impulse_end_idx = int(i_end)
        self.strength = float(strength)

def detect_impulse_and_zone(ohlcv: np.ndarray, cfg: dict) -> Optional[Zone]:
    if ohlcv.shape[0] < max(50, cfg["impulse"]["atr_len"] * 4):
        return None
    o = ohlcv[:, 1]; h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]
    atr_len = cfg["impulse"]["atr_len"]; min_body = cfg["impulse"]["body_min"]
    atr_mult = cfg["impulse"]["atr_mult"]; min_consec = cfg["impulse"]["min_consecutive"]

    # last ATR (for size tests)
    cur_atr = atr_last(ohlcv, atr_len)

    run_dir = 0; run_len = 0; end_idx = None
    for i in range(len(ohlcv) - 2, atr_len, -1):
        bod = candle_body_ratio(o[i], c[i], h[i], l[i])
        rng = h[i] - l[i]
        if bod >= min_body and rng >= atr_mult * cur_atr:
            d = 1 if c[i] > o[i] else -1
            if run_dir == 0 or d == run_dir:
                run_dir = d; run_len += 1
                if end_idx is None: end_idx = i
            else:
                break
            if run_len >= min_consec: break
        elif run_len > 0:
            break

    if end_idx is None or run_len < min_consec:
        return None

    i = end_idx
    if run_dir == 1:
        zone_top = float(l[i])
        body_low = float(min(o[i], c[i]))
        zone_bottom = float(min(zone_top, body_low))
        if zone_bottom > zone_top:
            zone_top, zone_bottom = zone_bottom, zone_top
        kind = "bullish"
    else:
        zone_bottom = float(h[i])
        body_high = float(max(o[i], c[i]))
        zone_top = float(max(zone_bottom, body_high))
        if zone_bottom > zone_top:
            zone_top, zone_bottom = zone_bottom, zone_top
        kind = "bearish"

    impulse_range = abs(h[i] - l[i])
    max_pct = cfg["zones"]["max_zone_pct"]
    max_thick = max_pct * max(impulse_range, 1e-9)
    thickness = abs(zone_top - zone_bottom)
    if thickness > max_thick and thickness > 0:
        mid = (zone_top + zone_bottom) / 2.0
        zone_top = mid + max_thick / 2.0
        zone_bottom = mid - max_thick / 2.0

    strength = min(1.0, candle_body_ratio(o[i], c[i], h[i], l[i]) * (impulse_range / max(cur_atr, 1e-9)))
    return Zone("bullish" if run_dir == 1 else "bearish",
                float(max(zone_top, zone_bottom)), float(min(zone_top, zone_bottom)),
                int(i), float(strength))

def within(x: float, a: float, b: float) -> bool:
    lo, hi = min(a, b), max(a, b)
    return lo - 1e-9 <= x <= hi + 1e-9

def check_bos_sfp(ohlcv: np.ndarray, zone: Zone, cfg: dict) -> Optional[Dict]:
    confirm = cfg["signals"]["confirm_closes"]
    sfp_w = cfg["signals"]["sfp_window"]
    h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]

    if zone.kind == "bullish":
        closes = [c[-k] for k in range(1, confirm + 1)]
        if all(v <= zone.bottom for v in closes):
            return {"type": "BOS", "direction": "bearish", "level": zone.bottom}
        for k in range(1, min(sfp_w, len(ohlcv) - 1) + 1):
            if l[-k] < zone.bottom and within(c[-k], zone.bottom, zone.top):
                return {"type": "SFP", "direction": "bullish", "level": zone.bottom}
    else:
        closes = [c[-k] for k in range(1, confirm + 1)]
        if all(v >= zone.top for v in closes):
            return {"type": "BOS", "direction": "bullish", "level": zone.top}
        for k in range(1, min(sfp_w, len(ohlcv) - 1) + 1):
            if h[-k] > zone.top and within(c[-k], zone.bottom, zone.top):
                return {"type": "SFP", "direction": "bearish", "level": zone.top}
    return None

def build_plan(ohlcv: np.ndarray, zone: Zone, sig: Dict, cfg: dict) -> Dict:
    cur_atr = atr_last(ohlcv, cfg["impulse"]["atr_len"])
    width = abs(zone.top - zone.bottom)

    off_cfg = cfg["risk"]["retest_offset_pct"]
    if isinstance(off_cfg, str) and off_cfg.lower() == "auto":
        frac = max(0.05, min(0.25, width / max(4 * cur_atr, 1e-9)))  # 5–25% of zone
    else:
        frac = float(off_cfg)

    pad_mult = float(cfg["risk"]["stop_atr_mult"])
    rr_mult = float(cfg["risk"]["tp_rr"])

    if sig["direction"] == "bullish":
        entry = float(zone.bottom + frac * width)
        stop = float(zone.bottom - pad_mult * cur_atr)
        risk = max(abs(entry - stop), 1e-9)
        tp1 = entry + rr_mult * risk
    else:
        entry = float(zone.top - frac * width)
        stop = float(zone.top + pad_mult * cur_atr)
        risk = max(abs(stop - entry), 1e-9)
        tp1 = entry - rr_mult * risk

    return {"entry": entry, "stop": stop, "tp1": tp1, "atr": float(cur_atr)}

def regime_tag_from_vol(ohlcv: np.ndarray, cfg: dict) -> str:
    h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]
    long_atr = atr_last(ohlcv, cfg["regime"]["atr_ma_len"])
    rng = np.mean(h[-50:] - l[-50:])
    return "trending" if (rng / max(long_atr, 1e-9)) >= cfg["regime"]["trend_ratio_min"] else "ranging"

def htf_bias_simple(ohlcv_htf: np.ndarray) -> Optional[str]:
    if ohlcv_htf.shape[0] < 20:
        return None
    closes = ohlcv_htf[:, 4]
    ma = np.mean(closes[-20:])
    return "bullish" if closes[-1] >= ma else "bearish"

def score_signal(zone: Zone, sig: Dict, plan: Dict, cfg: dict,
                 regime_tag: str, htf_bias: Optional[str]) -> float:
    # same components as live
    score = 0.0
    zs = min(1.0, max(0.0, zone.strength))
    score += 100 * zs * cfg["scoring"]["w_zone_strength"]
    dist = abs(plan["entry"] - sig["level"])
    clean = max(0.0, 1.0 - dist / max(plan["atr"], 1e-9))
    score += 100 * clean * cfg["scoring"]["w_signal_clean"]
    if (sig["type"] == "BOS" and regime_tag == "trending") or (sig["type"] == "SFP" and regime_tag == "ranging"):
        score += 100 * cfg["scoring"]["w_regime"]
    if cfg["filters"]["use_htf_bias"] and htf_bias:
        score += 100 * (1.0 if htf_bias == sig["direction"] else 0.0) * cfg["scoring"]["w_bias"]
    return float(score)

# ---------- outcome simulation ----------

def simulate_forward(df: pd.DataFrame, start_i: int, direction: str,
                     entry: float, stop: float, tp1: float) -> Tuple[str, float, int, Optional[int], Optional[int], Optional[float]]:
    """
    Walk forward bars:
      - wait for limit entry touch
      - after entry, SL takes priority if same bar hits both
    Returns: (outcome, rr, minutes_in_trade, enter_t, exit_t, exit_price)
    """
    enter_t = None
    def touched(lo, hi, px) -> bool:
        return lo <= px <= hi

    rr = abs((tp1 - entry) / (entry - stop)) if (entry - stop) != 0 else 0.0

    for i in range(start_i + 1, len(df)):
        row = df.iloc[i]
        low, high, t = float(row["l"]), float(row["h"]), int(row["t"])

        if enter_t is None:
            if touched(low, high, entry):
                enter_t = t
                if direction == "bullish":
                    if low <= stop:   # SL priority on the entry bar
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
    return "open", 0.0, int((int(df.iloc[-1]['t']) - enter_t) / 60000), enter_t, None, None

# ---------- filters (hard) ----------

def allowed_by_filters(cfg: dict, symbol: str, tf: str, sig_type: str, when_utc: dt.datetime,
                       atr_ser: np.ndarray) -> Tuple[bool, str]:
    min_pct = int(cfg["filters"].get("min_atr_percentile", 0))
    atr_pct = percentile_of_last(atr_ser, 200)
    if atr_pct < min_pct:
        return False, f"atr_pct {atr_pct:.0f} < {min_pct}"

    allow = cfg["filters"].get("allowed", {}).get(symbol, {}).get(tf, None)
    if allow:
        if "types" in allow and sig_type not in allow["types"]:
            return False, f"type {sig_type} not allowed"
        if "hours_utc" in allow and when_utc.hour not in set(allow["hours_utc"]):
            return False, f"hour {when_utc.hour} not allowed"
    return True, ""

# ---------- main runner ----------

def run_backtest(cfg: dict, args):
    research = dict(cfg.get("research", {}))

    exchange = args.exchange or research.get("exchange", "binance")
    symbols = args.symbols or research.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframes = args.timeframes or research.get("timeframes", ["1m"])
    days = int(args.days or research.get("days", 30))
    min_score = float(args.min_score or research.get("min_score", 90))
    cache_dir = args.cache or research.get("cache_dir", "cache")
    push_db = True if args.push_db is None else args.push_db
    save_csv = True if args.save_csv is None else args.save_csv

    signals_table = (research.get("signals_table")
                     or cfg.get("supabase", {}).get("signals_table")
                     or os.getenv("SUPABASE_TABLE", "signals"))

    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    supa = Supa(sb_url, sb_key, signals_table) if (push_db and sb_url and sb_key) else None
    if push_db and not supa:
        print("[DB] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY — skipping DB writes")

    ex_name = exchange
    print(f"[RESEARCH] exchange={ex_name} symbols={symbols} tfs={timeframes} days={days} min_score={min_score}")

    until = dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(days=days)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)

    for sym in symbols:
        for tf in timeframes:
            print(f"[RUN] {sym} {tf} … fetch/calc")
            try:
                df = fetch_ohlcv_cached(sym, tf, since_ms, until_ms, cache_dir, ex_name)
            except Exception as e:
                print(f"[ERR] fetch {sym} {tf}: {e}")
                continue

            if df.empty or len(df) < 320:
                print(f"[WARN] no/insufficient data for {sym} {tf}")
                continue

            arr = df[["t","o","h","l","c","v"]].to_numpy(dtype=float)
            out_rows: List[dict] = []
            window = max(300, 80)

            # Precompute ATR series once
            atr_ser = atr_series(arr, cfg["impulse"]["atr_len"])

            for i in range(window, len(df) - 1):
                sub = arr[i - window:i + 1]
                zone = detect_impulse_and_zone(sub, cfg)
                if not zone:
                    continue
                sig = check_bos_sfp(sub, zone, cfg)
                if not sig:
                    continue

                # HARD regime gating (align with live)
                regime = regime_tag_from_vol(sub, cfg)
                if (sig["type"] == "BOS" and regime != "trending") or (sig["type"] == "SFP" and regime != "ranging"):
                    continue

                # ATR percentile + allowed types/hours (UTC)
                when_utc = dt.datetime.fromtimestamp(int(sub[-1,0]) / 1000.0, tz=dt.timezone.utc)
                ok, why = allowed_by_filters(cfg, sym, tf, sig["type"], when_utc, atr_ser[:i+1])
                if not ok:
                    continue

                plan = build_plan(sub, zone, sig, cfg)

                # Optional HTF bias
                bias = None
                if cfg["filters"].get("use_htf_bias", False):
                    try:
                        ex = get_exchange(ex_name)
                        htf_tf = cfg["filters"]["htf_timeframe"]
                        htf = ex.fetch_ohlcv(sym, htf_tf, limit=200)
                        bias = htf_bias_simple(np.array(htf, dtype=float))
                    except Exception:
                        bias = None

                score = score_signal(zone, sig, plan, cfg, regime, bias)
                gate_tf = cfg["scoring"]["tf_overrides"].get(tf, cfg["scoring"]["min_score_to_alert"])
                if score < max(min_score, gate_tf):
                    continue

                outcome, rr, mins, enter_t, exit_t, exit_px = simulate_forward(
                    df, i, sig["direction"], plan["entry"], plan["stop"], plan["tp1"]
                )

                created_ts = dt.datetime.fromtimestamp(int(arr[i,0]) / 1000.0, tz=dt.timezone.utc)
                row = {
                    "created_at": created_ts.isoformat(),
                    "symbol": sym, "timeframe": tf,
                    "type": sig["type"], "direction": sig["direction"],
                    "zone_kind": "bullish" if sig["direction"] == "bullish" else "bearish",
                    "zone_top": float(zone.top), "zone_bottom": float(zone.bottom),
                    "level": float(sig["level"]), "idx": int(i),
                    "entry": float(plan["entry"]), "stop": float(plan["stop"]), "tp1": float(plan["tp1"]),
                    "atr": float(plan["atr"]), "score": float(score),
                    "reasons": f"{regime}|bias:{bias or 'n/a'}",
                    "is_backtest": True,
                    "dedupe_key": f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],6)}:{int(arr[i,0])}",
                    "entered_at": None if enter_t is None else dt.datetime.fromtimestamp(enter_t / 1000.0, tz=dt.timezone.utc).isoformat(),
                    "exited_at": None if exit_t is None else dt.datetime.fromtimestamp(exit_t / 1000.0, tz=dt.timezone.utc).isoformat(),
                    "exit_price": exit_px,
                    "rr_achieved": rr,
                    "outcome": outcome,
                    "minutes_in_trade": int(mins),
                }
                out_rows.append(row)

                # periodic flush
                if supa and len(out_rows) >= 500:
                    supa.insert(out_rows)
                    out_rows = []

            # final flush
            if supa and out_rows:
                supa.insert(out_rows)

            if save_csv:
                ensure_dir("backtests")
                outfile = pathlib.Path("backtests") / f"{sym.replace('/','-')}_{tf}_{days}d.csv"
                pd.DataFrame(out_rows).to_csv(outfile, index=False)
                print(f"[SAVE] {outfile} ({len(out_rows)} rows)")

    print("[DONE] research run complete.")

# ---------- CLI ----------

def build_argparser():
    p = argparse.ArgumentParser(description="StructureBot v2 — Research Backtester")
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
