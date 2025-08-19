# run_structure_bot.py
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import ccxt
import httpx
import numpy as np
import yaml

from structurebot.db import DB


CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")
WEBHOOK_ENV = os.getenv("DISCORD_WEBHOOK_URL", "")

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "12h": 43_200_000, "1d": 86_400_000
}

# ------------------- helpers -------------------

def load_cfg() -> dict:
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def resolve_symbol(ex: ccxt.Exchange, sym: str) -> str:
    """Handle Bybit symbol variants (e.g., BTC/USDT -> BTC/USDT:USDT)."""
    try:
        mkts = ex.load_markets()
        if sym in mkts: return sym
        if sym.endswith("/USDT") and f"{sym}:USDT" in mkts: return f"{sym}:USDT"
        if sym.endswith(":USDT") and sym.replace(":USDT","") in mkts: return sym.replace(":USDT","")
    except Exception:
        pass
    return sym

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> float:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    alpha = 1.0 / length
    rma = 0.0
    for v in tr[-length * 4:]:
        rma = alpha * v + (1 - alpha) * rma
    return float(rma)

def series_atr(arr: np.ndarray, length: int = 14) -> np.ndarray:
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

def percentile_of_last(values: np.ndarray, lookback: int = 200) -> float:
    if len(values) < lookback:
        lookback = len(values)
    window = values[-lookback:]
    last = window[-1]
    rank = (window <= last).mean() * 100.0
    return float(rank)

@dataclass
class Zone:
    kind: str           # 'bullish' | 'bearish'
    top: float
    bottom: float
    impulse_end_idx: int
    strength: float     # 0..1

def candle_body_ratio(o: float, c: float, h: float, l: float) -> float:
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    return float(body / rng)

def detect_impulse_and_zone(ohlcv: np.ndarray, cfg: dict) -> Optional[Zone]:
    if ohlcv.shape[0] < max(50, cfg["impulse"]["atr_len"] * 4):
        return None
    o = ohlcv[:, 1]; h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]
    atr_len   = cfg["impulse"]["atr_len"]
    min_body  = cfg["impulse"]["body_min"]
    atr_mult  = cfg["impulse"]["atr_mult"]
    min_consec= cfg["impulse"]["min_consecutive"]

    cur_atr = atr(h, l, c, atr_len)

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

    i = end_idx; up = (run_dir == 1)
    if up:
        zone_top    = float(l[i])
        body_low    = float(min(o[i], c[i]))
        zone_bottom = float(min(zone_top, body_low))
        if zone_bottom > zone_top: zone_top, zone_bottom = zone_bottom, zone_top
        kind = "bullish"
    else:
        zone_bottom = float(h[i])
        body_high   = float(max(o[i], c[i]))
        zone_top    = float(max(zone_bottom, body_high))
        if zone_bottom > zone_top: zone_top, zone_bottom = zone_bottom, zone_top
        kind = "bearish"

    # clamp zone thickness
    impulse_range = abs(h[i] - l[i])
    max_pct   = cfg["zones"]["max_zone_pct"]
    max_thick = max_pct * max(impulse_range, 1e-9)
    thickness = abs(zone_top - zone_bottom)
    if thickness > max_thick and thickness > 0:
        mid = (zone_top + zone_bottom) / 2.0
        zone_top = mid + max_thick / 2.0
        zone_bottom = mid - max_thick / 2.0

    strength = min(1.0, candle_body_ratio(o[i], c[i], h[i], l[i]) * (impulse_range / max(cur_atr, 1e-9)))
    return Zone(kind=kind, top=float(max(zone_top, zone_bottom)), bottom=float(min(zone_top, zone_bottom)),
                impulse_end_idx=int(i), strength=float(strength))

def within(x: float, a: float, b: float) -> bool:
    lo, hi = min(a, b), max(a, b)
    return lo - 1e-9 <= x <= hi + 1e-9

def check_bos_sfp(ohlcv: np.ndarray, zone: Zone, cfg: dict) -> Optional[Dict]:
    confirm = cfg["signals"]["confirm_closes"]
    sfp_w   = cfg["signals"]["sfp_window"]
    h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]

    if zone.kind == "bullish":
        closes = [c[-k] for k in range(1, min(confirm, len(ohlcv)-1) + 1)]
        if closes and all(val <= zone.bottom for val in closes):
            return {"type": "BOS", "direction": "bearish", "level": zone.bottom}
        for k in range(1, min(sfp_w, len(ohlcv) - 1) + 1):
            if l[-k] < zone.bottom and within(c[-k], zone.bottom, zone.top):
                return {"type": "SFP", "direction": "bullish", "level": zone.bottom}
    else:
        closes = [c[-k] for k in range(1, min(confirm, len(ohlcv)-1) + 1)]
        if closes and all(val >= zone.top for val in closes):
            return {"type": "BOS", "direction": "bullish", "level": zone.top}
        for k in range(1, min(sfp_w, len(ohlcv) - 1) + 1):
            if h[-k] > zone.top and within(c[-k], zone.bottom, zone.top):
                return {"type": "SFP", "direction": "bearish", "level": zone.top}
    return None

def build_plan(ohlcv: np.ndarray, zone: Zone, sig: Dict, cfg: dict) -> Dict:
    h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]
    cur_atr = atr(h, l, c, cfg["impulse"]["atr_len"])

    width  = abs(zone.top - zone.bottom)
    off_cfg = cfg["risk"]["retest_offset_pct"]
    if isinstance(off_cfg, str) and off_cfg.lower() == "auto":
        # deeper when zones are wide vs ATR, clamp 5%..25% (width / (4*ATR) heuristic)
        frac = max(0.05, min(0.25, width / max(4 * cur_atr, 1e-9)))
    else:
        frac = float(off_cfg)

    pad_mult = float(cfg["risk"]["stop_atr_mult"])
    rr_mult  = float(cfg["risk"]["tp_rr"])

    if sig["direction"] == "bullish":
        entry = float(zone.bottom + frac * width)
        stop  = float(zone.bottom - pad_mult * cur_atr)
        risk  = max(abs(entry - stop), 1e-9)
        tp1   = entry + rr_mult * risk
    else:
        entry = float(zone.top - frac * width)
        stop  = float(zone.top + pad_mult * cur_atr)
        risk  = max(abs(stop - entry), 1e-9)
        tp1   = entry - rr_mult * risk

    return {"entry": entry, "stop": stop, "tp1": tp1, "atr": float(cur_atr)}

def regime_tag_from_vol(ohlcv: np.ndarray, cfg: dict) -> str:
    h = ohlcv[:, 2]; l = ohlcv[:, 3]
    c = ohlcv[:, 4]
    long_atr = atr(h, l, c, cfg["regime"]["atr_ma_len"])
    rng = np.mean(h[-50:] - l[-50:])
    return "trending" if (rng / max(long_atr, 1e-9)) >= cfg["regime"]["trend_ratio_min"] else "ranging"

def htf_bias_simple(ohlcv_htf: np.ndarray) -> Optional[str]:
    if ohlcv_htf.shape[0] < 20:
        return None
    closes = ohlcv_htf[:, 4]
    ma = np.mean(closes[-20:])
    return "bullish" if closes[-1] >= ma else "bearish"

def score_signal(zone: Zone, sig: Dict, plan: Dict, cfg: dict,
                 regime_tag: str, htf_bias: Optional[str]) -> Tuple[float, List[str]]:
    reasons = []; score = 0.0
    zs = min(1.0, max(0.0, zone.strength))
    score += 100 * zs * cfg["scoring"]["w_zone_strength"]; reasons.append(f"zone:{zs:.2f}")
    dist = abs(plan["entry"] - sig["level"]); clean = max(0.0, 1.0 - dist / max(plan["atr"], 1e-9))
    score += 100 * clean * cfg["scoring"]["w_signal_clean"]; reasons.append(f"clean:{clean:.2f}")
    if (sig["type"] == "BOS" and regime_tag == "trending") or (sig["type"] == "SFP" and regime_tag == "ranging"):
        score += 100 * 1.0 * cfg["scoring"]["w_regime"]; reasons.append(f"reg:{regime_tag}✓")
    else:
        reasons.append(f"reg:{regime_tag}×")
    if cfg["filters"]["use_htf_bias"] and htf_bias:
        ok = (htf_bias == sig["direction"])
        score += 100 * (1.0 if ok else 0.0) * cfg["scoring"]["w_bias"]
        reasons.append(f"bias:{htf_bias}{'✓' if ok else '×'}")
    return score, reasons

def post_discord(webhook: str, title: str, fields: List[Dict], footer: str, color: int = 0x5865F2):
    if not webhook: return
    payload = {"username": "StructureBot","embeds": [{"title": title,"color": color,"fields": fields,"footer": {"text": footer}}]}
    try:
        with httpx.Client(timeout=15) as cli:
            cli.post(webhook, json=payload)
    except Exception as e:
        print(f"[DISCORD] post error: {e}")

def allowed_by_filters(cfg: dict, symbol: str, tf: str, sig_type: str, now_utc: datetime,
                       atr_series: np.ndarray) -> Tuple[bool, str]:
    # ATR percentile
    min_pct = int(cfg["filters"].get("min_atr_percentile", 0))
    atr_pct = percentile_of_last(atr_series, 200)
    if atr_pct < min_pct:
        return False, f"atr_pct {atr_pct:.0f} < {min_pct}"

    # Allowed lists
    allow = cfg["filters"].get("allowed", {}).get(symbol, {}).get(tf, None)
    if allow:
        if "types" in allow and sig_type not in allow["types"]:
            return False, f"type {sig_type} not allowed"
        if "hours_utc" in allow and now_utc.hour not in set(allow["hours_utc"]):
            return False, f"hour {now_utc.hour} not allowed"
    return True, ""

# ------------------- main -------------------

if __name__ == "__main__":
    cfg = load_cfg()
    webhook = cfg.get("discord_webhook_url") or WEBHOOK_ENV

    ex = ccxt.bybit({"enableRateLimit": True})
    db = DB()

    symbols: List[str] = cfg["symbols"]
    tfs: List[str]     = cfg["timeframes"]
    lb = int(cfg["lookback_bars"])
    poll = int(cfg["poll_seconds"])

    # memory de-dupe for alerts
    seen: deque[Tuple[str, datetime]] = deque(maxlen=2048)
    dedupe_minutes = int(cfg.get("dedupe_minutes", 30))

    # -------- startup backfill --------
    if cfg.get("startup_backfill", {}).get("enabled", True):
        bf_bars = int(cfg["startup_backfill"]["lookback_bars"])
        max_per = int(cfg["startup_backfill"]["max_signals_per_market"])
        print(f"[BACKFILL] bars={bf_bars}, max_per_market={max_per}")

        for sym in symbols:
            for tf in tfs:
                try:
                    rsym = resolve_symbol(ex, sym)
                    ohlcv = ex.fetch_ohlcv(rsym, tf, limit=min(1500, bf_bars))
                except Exception as e:
                    print(f"[BACKFILL_ERR] {sym} {tf}: {e}")
                    continue
                if not ohlcv:
                    continue
                arr = np.array(ohlcv, dtype=float)
                zone = detect_impulse_and_zone(arr, cfg)
                if not zone: continue
                sig = check_bos_sfp(arr, zone, cfg)
                if not sig:
                   continue

                # Hard regime filter
                regime = regime_tag_from_vol(arr, cfg)
                if (sig["type"] == "BOS" and regime != "trending") or (sig["type"] == "SFP" and regime != "ranging"):
                    continue

                atr_ser = series_atr(arr, cfg["impulse"]["atr_len"])
                ok, _ = allowed_by_filters(cfg, sym, tf, sig["type"], datetime.now(timezone.utc), atr_ser)
                if not ok: continue

                plan = build_plan(arr, zone, sig, cfg)
                bias = None
                if cfg["filters"]["use_htf_bias"]:
                    htf_tf = cfg["filters"]["htf_timeframe"]
                    try:
                        htf = ex.fetch_ohlcv(rsym, htf_tf, limit=200)
                        bias = htf_bias_simple(np.array(htf, dtype=float))
                    except Exception:
                        bias = None

                score, reasons = score_signal(zone, sig, plan, cfg, regime, bias)
                gate_tf = cfg["scoring"]["tf_overrides"].get(tf, cfg["scoring"]["min_score_to_alert"])
                if score < gate_tf: continue

                key = f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(float(sig['level']), 2)}"
                if any(k == key and (datetime.now(timezone.utc)-t).total_seconds() < dedupe_minutes*60 for k,t in seen):
                    continue
                seen.append((key, datetime.now(timezone.utc)))

                title = f"RECENT {sig['type']} — {sym} {tf} ({sig['direction'].upper()})"
                fields = [
                    {"name":"Zone","value": f"{zone.kind} | Level: {sig['level']:.2f}", "inline":False},
                    {"name":"Entry (limit)","value": f"{plan['entry']:.2f}", "inline":True},
                    {"name":"Stop","value": f"{plan['stop']:.2f}", "inline":True},
                    {"name":f"TP1 (~{cfg['risk']['tp_rr']}R)","value": f"{plan['tp1']:.2f}", "inline":True},
                    {"name":"ATR","value": f"{plan['atr']:.2f}", "inline":True},
                    {"name":"Score","value": f"{score:.1f} / 100", "inline":True},
                ]
                post_discord(webhook, title, fields, footer=f"{regime} • filters ok")
                db.log_signal(sym, tf, sig, zone, plan, score, reasons, False, key)

    print(f"[INFO] symbols={symbols} tfs={tfs}")
    last_hb = datetime.now(timezone.utc); HEART_MIN = int(cfg["debug"].get("heartbeat_minutes", 2))

    # ---------------- main loop ----------------
    while True:
        for sym in symbols:
            for tf in tfs:
                print(f"[SCAN] {sym} {tf} …")
                try:
                    rsym = resolve_symbol(ex, sym)
                    ohlcv = ex.fetch_ohlcv(rsym, tf, limit=min(lb, 1500))
                except Exception as e:
                    print(f"[ERR] {sym} {tf}: {e}"); continue
                if not ohlcv or len(ohlcv) < 80:
                    print(f"[WAIT] {sym} {tf} — insufficient data"); continue

                arr = np.array(ohlcv, dtype=float)
                zone = detect_impulse_and_zone(arr, cfg);  if not zone: continue
                sig  = check_bos_sfp(arr, zone, cfg);       if not sig: continue

                regime = regime_tag_from_vol(arr, cfg)
                if (sig["type"] == "BOS" and regime != "trending") or (sig["type"] == "SFP" and regime != "ranging"):
                    continue

                atr_ser = series_atr(arr, cfg["impulse"]["atr_len"])
                ok, why = allowed_by_filters(cfg, sym, tf, sig["type"], datetime.now(timezone.utc), atr_ser)
                if not ok:
                    if cfg["debug"].get("log_scans", False):
                        print(f"[FILTER] {sym} {tf} {sig['type']} skip — {why}")
                    continue

                plan  = build_plan(arr, zone, sig, cfg)
                bias  = None
                if cfg["filters"]["use_htf_bias"]:
                    htf_tf = cfg["filters"]["htf_timeframe"]
                    try:
                        htf = ex.fetch_ohlcv(rsym, htf_tf, limit=200)
                        bias = htf_bias_simple(np.array(htf, dtype=float))
                    except Exception:
                        bias = None

                score, reasons = score_signal(zone, sig, plan, cfg, regime, bias)
                gate_tf = cfg["scoring"]["tf_overrides"].get(tf, cfg["scoring"]["min_score_to_alert"])
                if score < gate_tf:
                    if cfg["debug"].get("log_scans", False):
                        print(f"[FILTER] {sym} {tf} {sig['type']} score {score:.1f} < gate {gate_tf}")
                    continue

                key = f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(float(sig['level']), 2)}"
                now = datetime.now(timezone.utc)
                # prune & dedupe memory
                new_seen = deque(maxlen=2048)
                while seen:
                    k, t = seen.popleft()
                    if (now - t).total_seconds() < dedupe_minutes * 60:
                        new_seen.append((k, t))
                seen = new_seen
                if any(k == key for k, _ in seen):
                    continue
                seen.append((key, now))

                title = f"{sig['type']} — {sym} {tf} ({sig['direction'].upper()})"
                fields = [
                    {"name":"Zone","value": f"{zone.kind} | Level: {sig['level']:.2f}", "inline":False},
                    {"name":"Close","value": f"{arr[-1,4]:.2f} | Time (ms): {int(arr[-1,0])}", "inline":False},
                    {"name":"Entry (limit)","value": f"{plan['entry']:.2f}", "inline":True},
                    {"name":"Stop","value": f"{plan['stop']:.2f}", "inline":True},
                    {"name":f"TP1 (~{cfg['risk']['tp_rr']}R)","value": f"{plan['tp1']:.2f}", "inline":True},
                    {"name":"ATR","value": f"{plan['atr']:.2f}", "inline":True},
                    {"name":"Score","value": f"{score:.1f} / 100", "inline":True},
                ]
                post_discord(webhook, title, fields, footer=f"{regime} • filters ok",
                             color=0x2ecc71 if sig["direction"] == "bullish" else 0xe74c3c)

                db.log_signal(sym, tf, sig, zone, plan, score, reasons, False, key)

        if (datetime.now(timezone.utc) - last_hb).total_seconds() >= HEART_MIN * 60:
            print(f"[HEARTBEAT] {datetime.now(timezone.utc).isoformat()} — sleeping {poll}s")
            last_hb = datetime.now(timezone.utc)
        time.sleep(poll)
