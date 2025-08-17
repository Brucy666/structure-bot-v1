#!/usr/bin/env python3
import os, sys, time, math, json, pathlib, argparse, datetime as dt
from typing import List, Dict, Optional, Tuple
import yaml
import requests

import pandas as pd
import numpy as np

# ---- optional: CCXT for historical candles (Binance by default for reliability) ----
try:
    import ccxt
except Exception:
    ccxt = None


# -----------------------------
# Helpers
# -----------------------------
def utc_ms(ts: dt.datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def load_cfg() -> dict:
    # Use STRUCTURE_CONFIG env first, fall back to config.yml
    s = os.environ.get("STRUCTURE_CONFIG")
    if s:
        return yaml.safe_load(s)
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Simple Supabase PostgREST client (inserts)
# -----------------------------
class Supa:
    def __init__(self, url: str, key: str, table_signals: str):
        self.url = url.rstrip("/")
        self.key = key
        self.table_signals = table_signals
        self.s = requests.Session()
        self.s.headers.update({
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        })

    def insert_signals(self, rows: List[dict]) -> None:
        if not rows:
            return
        endpoint = f"{self.url}/rest/v1/{self.table_signals}"
        r = self.s.post(endpoint, data=json.dumps(rows))
        if r.status_code >= 300:
            print(f"[DB][ERR] insert_signals {r.status_code}: {r.text}", file=sys.stderr)

    def patch_signal(self, dedupe_key: str, updates: dict) -> None:
        endpoint = f"{self.url}/rest/v1/{self.table_signals}"
        r = self.s.patch(
            endpoint,
            params={"select": "*", "dedupe_key": f"eq.{dedupe_key}"},
            data=json.dumps(updates)
        )
        if r.status_code >= 300:
            print(f"[DB][ERR] patch_signal {r.status_code}: {r.text}", file=sys.stderr)


# -----------------------------
# Candle fetching (cache + CCXT)
# -----------------------------
def get_exchange(name: str):
    if ccxt is None:
        raise RuntimeError("ccxt is not installed. Add it to requirements.txt")
    name = (name or "binance").lower()
    klass = getattr(ccxt, name)
    ex = klass({"enableRateLimit": True})
    # Default to USDT futures when available for better continuity
    if hasattr(ex, "options"):
        ex.options.setdefault("defaultType", "future")
    return ex

def fetch_ohlcv_cached(symbol: str, timeframe: str, since_ms: int, until_ms: int,
                       cache_dir: str, exchange_name: str = "binance") -> pd.DataFrame:
    ensure_dir(cache_dir)
    ex = get_exchange(exchange_name)

    cache_file = pathlib.Path(cache_dir) / f"{exchange_name}_{symbol.replace('/', '-')}_{timeframe}.parquet"
    df_cached = None
    if cache_file.exists():
        try:
            df_cached = pd.read_parquet(cache_file)
        except Exception:
            df_cached = None

    # pull from exchange if needed
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    needed_from = since_ms if df_cached is None else int(df_cached["t"].iloc[-1]) + tf_ms

    all_rows = [] if df_cached is None else df_cached.to_dict("records")

    if needed_from <= until_ms:
        fetch_from = needed_from
        while fetch_from < until_ms:
            limit = 1000
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_from, limit=limit)
            if not batch:
                break
            for o in batch:
                # [t, o, h, l, c, v]
                if o[0] > until_ms:
                    break
                all_rows.append({"t": int(o[0]), "o": float(o[1]), "h": float(o[2]),
                                 "l": float(o[3]), "c": float(o[4]), "v": float(o[5])})
            fetch_from = batch[-1][0] + tf_ms
            # rate-limit protection
            time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(all_rows).drop_duplicates("t").sort_values("t")
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


# -----------------------------
# Strategy / Structure detection (compact + consistent with live logic)
# -----------------------------
def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = series_close.shift(1)
    tr = pd.concat([
        series_high - series_low,
        (series_high - prev_close).abs(),
        (series_low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def pivots(h: pd.Series, l: pd.Series, left: int = 2, right: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Return boolean series for swing highs/lows."""
    sh = (h.shift(left).rolling(left + right + 1).max() == h) & (h == h.rolling(left + right + 1, center=True).max())
    sl = (l.shift(left).rolling(left + right + 1).min() == l) & (l == l.rolling(left + right + 1, center=True).min())
    return sh.fillna(False), sl.fillna(False)

def regime(close: pd.Series, lookback: int = 50) -> str:
    if len(close) < lookback + 5:
        return "ranging"
    x = np.arange(lookback)
    y = close.iloc[-lookback:].values
    # simple slope via linear fit
    slope = np.polyfit(x, y, 1)[0]
    return "trending" if abs(slope) > (np.std(y) * 0.002) else "ranging"

def bias(close: pd.Series, lookback: int = 50) -> str:
    if len(close) < lookback:
        return "neutral"
    return "bullish" if close.iloc[-1] > close.rolling(lookback).mean().iloc[-1] else "bearish"

def detect_bos_sfp(df: pd.DataFrame, tf: str) -> Optional[Dict]:
    """
    Very compact detector:
    - Find fresh pivot + break (BOS) or wick-through + close-back (SFP)
    - Build wick->body zone from last impulse bar
    """
    if len(df) < 80:
        return None
    o, h, l, c = df.o, df.h, df.l, df.c

    # swings
    sh, sl = pivots(h, l, left=2, right=2)
    idx = df.index[-1]
    last_20 = df.iloc[-60:]

    # check for SFP at the latest bar (wick through prev swing, close back in)
    sfp_type, level = None, None
    # last swing high/low in lookback
    last_swing_high = last_20[sh].h.max() if (last_20[sh].shape[0] > 0) else None
    last_swing_low  = last_20[sl].l.min() if (last_20[sl].shape[0] > 0) else None

    if last_swing_high is not None and h.iloc[-1] > last_swing_high and c.iloc[-1] < last_swing_high:
        sfp_type, level = ("bearish", float(last_swing_high))
    if last_swing_low is not None and l.iloc[-1] < last_swing_low and c.iloc[-1] > last_swing_low:
        sfp_type, level = ("bullish", float(last_swing_low)) if sfp_type is None else sfp_type

    # check for BOS (body close breaking last swing)
    bos_type, bos_level = None, None
    if last_swing_high is not None and c.iloc[-1] > last_swing_high:
        bos_type, bos_level = ("bullish", float(last_swing_high))
    if last_swing_low is not None and c.iloc[-1] < last_swing_low:
        bos_type, bos_level = ("bearish", float(last_swing_low)) if bos_type is None else bos_type

    atr_val = atr(h, l, c, 14).iloc[-1]
    reg = regime(c)
    b = bias(c)

    # prefer BOS over SFP if both present
    if bos_type:
        direction = "bullish" if bos_type == "bullish" else "bearish"
        # zone = wick->body on the impulse bar (use previous bar as impulse end)
        imp = df.iloc[-2]
        if direction == "bullish":
            z_top, z_bot = float(max(imp.o, imp.c)), float(min(imp.l, imp.o, imp.c))
        else:
            z_top, z_bot = float(max(imp.h, imp.o, imp.c)), float(min(imp.o, imp.c))
        entry = z_bot if direction == "bullish" else z_top
        stop  = entry - atr_val*0.8 if direction == "bullish" else entry + atr_val*0.8
        tp1   = entry + (entry - stop)*1.6 if direction == "bullish" else entry - (stop - entry)*1.6
        score = 80.0
        if reg == "trending": score += 10
        if (direction == "bullish" and b == "bullish") or (direction == "bearish" and b == "bearish"):
            score += 10
        clean = (abs(z_top - z_bot) / max(1e-8, atr_val))
        score += max(0.0, 10.0 - min(10.0, clean*2))
        return {
            "type": "BOS", "direction": direction,
            "zone_kind": "bearish" if direction == "bearish" else "bullish",
            "zone_top": z_top, "zone_bottom": z_bot,
            "level": float(bos_level),
            "entry": float(entry), "stop": float(stop), "tp1": float(tp1),
            "atr": float(atr_val), "score": round(score, 2),
            "reasons": f"bos•reg:{reg}•bias:{b}•zone:{abs(z_top-z_bot):.2f}↔atr:{atr_val:.2f}",
        }

    if sfp_type:
        direction = "bearish" if sfp_type == "bearish" else "bullish"
        imp = df.iloc[-2]
        if direction == "bullish":
            z_top, z_bot = float(max(imp.o, imp.c)), float(min(imp.l, imp.o, imp.c))
        else:
            z_top, z_bot = float(max(imp.h, imp.o, imp.c)), float(min(imp.o, imp.c))
        entry = z_bot if direction == "bullish" else z_top
        stop  = entry - atr_val*0.9 if direction == "bullish" else entry + atr_val*0.9
        tp1   = entry + (entry - stop)*1.6 if direction == "bullish" else entry - (stop - entry)*1.6
        score = 70.0
        if reg == "ranging": score += 10
        if (direction == "bullish" and b == "bullish") or (direction == "bearish" and b == "bearish"):
            score += 10
        clean = (abs(z_top - z_bot) / max(1e-8, atr_val))
        score += max(0.0, 10.0 - min(10.0, clean*2))
        return {
            "type": "SFP", "direction": direction,
            "zone_kind": "bearish" if direction == "bearish" else "bullish",
            "zone_top": z_top, "zone_bottom": z_bot,
            "level": float(level),
            "entry": float(entry), "stop": float(stop), "tp1": float(tp1),
            "atr": float(atr_val), "score": round(score, 2),
            "reasons": f"sfp•reg:{reg}•bias:{b}•zone:{abs(z_top-z_bot):.2f}↔atr:{atr_val:.2f}",
        }

    return None


# -----------------------------
# Simulate the trade outcome forward
# -----------------------------
def simulate_outcome(df: pd.DataFrame, start_idx: int, direction: str, entry: float, stop: float, tp1: float) -> Tuple[str, float, int, Optional[int], Optional[int], Optional[float]]:
    """
    Walk bars forward: returns (outcome, rr, minutes_in_trade, enter_t, exit_t, exit_price)
    """
    # wait for price to touch entry (limit)
    enter_t, exit_t, exit_px = None, None, None

    # Define hit functions
    def hit_long_entry(row):  return row["l"] <= entry <= row["h"]
    def hit_short_entry(row): return row["l"] <= entry <= row["h"]
    def hit_long_tp(row):     return row["h"] >= tp1
    def hit_long_sl(row):     return row["l"] <= stop
    def hit_short_tp(row):    return row["l"] <= tp1
    def hit_short_sl(row):    return row["h"] >= stop

    rr = abs((tp1 - entry) / (entry - stop)) if (entry - stop) != 0 else 0.0

    # from next bar onwards
    for i in range(start_idx+1, len(df)):
        row = df.iloc[i]
        # entry
        if enter_t is None:
            if (direction == "bullish" and hit_long_entry(row)) or (direction == "bearish" and hit_short_entry(row)):
                enter_t = int(row["t"])
                # after entry, check tp/sl in same bar with priority to worst-case adverse first
                if direction == "bullish":
                    if hit_long_sl(row):
                        exit_t, exit_px = int(row["t"]), stop
                        return "sl", -1.0, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px
                    if hit_long_tp(row):
                        exit_t, exit_px = int(row["t"]), tp1
                        return "tp", +rr, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px
                else:
                    if hit_short_sl(row):
                        exit_t, exit_px = int(row["t"]), stop
                        return "sl", -1.0, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px
                    if hit_short_tp(row):
                        exit_t, exit_px = int(row["t"]), tp1
                        return "tp", +rr, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px
        else:
            # already in trade, evaluate tp/sl
            if direction == "bullish":
                if hit_long_sl(row):
                    exit_t, exit_px = int(row["t"]), stop
                    return "sl", -1.0, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px
                if hit_long_tp(row):
                    exit_t, exit_px = int(row["t"]), tp1
                    return "tp", +rr, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px
            else:
                if hit_short_sl(row):
                    exit_t, exit_px = int(row["t"]), stop
                    return "sl", -1.0, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px
                if hit_short_tp(row):
                    exit_t, exit_px = int(row["t"]), tp1
                    return "tp", +rr, int((exit_t - enter_t)/60000), enter_t, exit_t, exit_px

    # never filled or never exited within window
    if enter_t is None:
        return "missed", 0.0, 0, None, None, None
    else:
        # open trade at end-of-window; mark as open/missed
        return "open", 0.0, int((int(df.iloc[-1]["t"]) - enter_t)/60000), enter_t, None, None


# -----------------------------
# Main backtest
# -----------------------------
def run_backtest(cfg: dict):
    research = cfg.get("research", {})
    symbols     = research.get("symbols", ["BTC/USDT","ETH/USDT","SOL/USDT"])
    timeframes  = research.get("timeframes", ["1m"])
    days        = int(research.get("days", 30))
    exchange    = research.get("exchange", "binance")
    cache_dir   = research.get("cache_dir", "cache")
    min_score   = float(research.get("min_score", 0))
    save_csv    = bool(research.get("save_csv", True))
    send_db     = bool(research.get("push_to_supabase", True))

    supa = None
    if send_db:
        sb_url = os.environ.get("SUPABASE_URL")
        sb_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        table  = research.get("signals_table", cfg.get("supabase", {}).get("signals_table", "signals"))
        if sb_url and sb_key:
            supa = Supa(sb_url, sb_key, table)
            print(f"[DB] Connected {sb_url} / table={table}")
        else:
            print("[DB] Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY — skipping DB writes")

    until = now_utc()
    since = until - dt.timedelta(days=days)
    since_ms, until_ms = utc_ms(since), utc_ms(until)

    for sym in symbols:
        for tf in timeframes:
            print(f"[RUN] {sym} {tf} … since {since.isoformat()} to {until.isoformat()}")
            df = fetch_ohlcv_cached(sym, tf, since_ms, until_ms, cache_dir, exchange)
            if df.empty:
                print(f"[WARN] no data for {sym} {tf}")
                continue
            # iterate bar-by-bar, detect signals, simulate outcome, collect rows
            rows = []
            window = 300  # sliding window
            for i in range(max(80, window), len(df)-1):
                sub = df.iloc[i-window:i+1].copy()
                sig = detect_bos_sfp(sub, tf)
                if not sig: 
                    continue
                if sig["score"] < min_score:
                    continue

                # simulate forward from i (signal bar at sub.tail(1))
                outcome, rr, mins, enter_t, exit_t, exit_px = simulate_outcome(df, i, sig["direction"], sig["entry"], sig["stop"], sig["tp1"])

                row = {
                    "created_at": dt.datetime.fromtimestamp(df.iloc[i]["t"]/1000.0, tz=dt.timezone.utc).isoformat(),
                    "symbol": sym, "timeframe": tf,
                    "type": sig["type"], "direction": sig["direction"],
                    "zone_kind": sig["zone_kind"],
                    "zone_top": round(sig["zone_top"], 8), "zone_bottom": round(sig["zone_bottom"], 8),
                    "level": round(sig["level"], 8), "idx": i,
                    "entry": round(sig["entry"], 8), "stop": round(sig["stop"], 8), "tp1": round(sig["tp1"], 8),
                    "atr": round(sig["atr"], 8), "score": float(sig["score"]),
                    "reasons": sig["reasons"], "is_backtest": True,
                    "dedupe_key": f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],6)}:{int(df.iloc[i]['t'])}",
                    "entered_at": None if enter_t is None else dt.datetime.fromtimestamp(enter_t/1000.0, tz=dt.timezone.utc).isoformat(),
                    "exited_at":  None if exit_t  is None else dt.datetime.fromtimestamp(exit_t/1000.0, tz=dt.timezone.utc).isoformat(),
                    "exit_price": exit_px, "rr_achieved": rr, "outcome": outcome,
                    "minutes_in_trade": mins
                }
                rows.append(row)

                # batch flush each 250 rows
                if supa and len(rows) >= 250:
                    supa.insert_signals(rows)
                    rows = []

            # final flush
            if supa and rows:
                supa.insert_signals(rows)

            if save_csv:
                out_dir = pathlib.Path("backtests")
                ensure_dir(out_dir)
                out_file = out_dir / f"{sym.replace('/','-')}_{tf}_{days}d.csv"
                pd.DataFrame(rows).to_csv(out_file, index=False)
                print(f"[SAVE] {out_file} ({len(rows)} rows)")

    print("[DONE] research run complete.")
    

if __name__ == "__main__":
    cfg = load_cfg()
    run_backtest(cfg)
