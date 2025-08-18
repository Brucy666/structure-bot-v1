# run_structure_bot.py (drop-in replacement)
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

def tf_ms(tf: str) -> int:
    return TF_MS.get(tf, 60_000)

def to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> float:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    alpha = 1.0 / length
    rma = 0.0
    for v in tr[-length * 4:]:  # warmup
        rma = alpha * v + (1 - alpha) * rma
    return float(rma)

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

    run_dir = 0
    run_len = 0
    end_idx = None

    for i in range(len(ohlcv) - 2, atr_len, -1):
        bod = candle_body_ratio(o[i], c[i], h[i], l[i])
        rng = h[i] - l[i]
        if bod >= min_body and rng >= atr_mult * cur_atr:
            d = 1 if c[i] > o[i] else -1
            if run_dir == 0 or d == run_dir:
                run_dir = d
                run_len += 1
                if end_idx is None:
                    end_idx = i
            else:
                break
            if run_len >= min_consec:
                break
        elif run_len > 0:
            break

    if end_idx is None or run_len < min_consec:
        return None

    i   = end_idx
    up  = (run_dir == 1)

    if up:
        zone_top    = float(l[i])
        body_low    = float(min(o[i], c[i]))
        zone_bottom = float(min(zone_top, body_low))
        if zone_bottom > zone_top:
            zone_top, zone_bottom = zone_bottom, zone_top
        kind = "bullish"
    else:
        zone_bottom = float(h[i])
        body_high   = float(max(o[i], c[i]))
        zone_top    = float(max(zone_bottom, body_high))
        if zone_bottom > zone_top:
            zone_top, zone_bottom = zone_bottom, zone_top
        kind = "bearish"

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

    if zone.kind == "bullish":
        closes = [ohlcv[-k, 4] for k in range(1, min(confirm, len(ohlcv)-1) + 1)]
        if closes and all(c <= zone.bottom for c in closes):
            return {"type": "BOS", "direction": "bearish", "level": zone.bottom, "idx": int(len(ohlcv) - 1)}
        for k in range(1, min(sfp_w, len(ohlcv) - 1) + 1):
            lo = ohlcv[-k, 3]; cl = ohlcv[-k, 4]
            if lo < zone.bottom and within(cl, zone.bottom, zone.top):
                return {"type": "SFP", "direction": "bullish", "level": zone.bottom, "idx": int(len(ohlcv) - k)}
    else:
        closes = [ohlcv[-k, 4] for k in range(1, min(confirm, len(ohlcv)-1) + 1)]
        if closes and all(c >= zone.top for c in closes):
            return {"type": "BOS", "direction": "bullish", "level": zone.top, "idx": int(len(ohlcv) - 1)}
        for k in range(1, min(sfp_w, len(ohlcv) - 1) + 1):
            hi = ohlcv[-k, 2]; cl = ohlcv[-k, 4]
            if hi > zone.top and within(cl, zone.bottom, zone.top):
                return {"type": "SFP", "direction": "bearish", "level": zone.top, "idx": int(len(ohlcv) - k)}

    return None

def build_plan(ohlcv: np.ndarray, zone: Zone, sig: Dict, cfg: dict) -> Dict:
    h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]
    cur_atr = atr(h, l, c, cfg["impulse"]["atr_len"])

    off      = cfg["risk"]["retest_offset_pct"]
    pad_mult = cfg["risk"]["stop_atr_mult"]
    rr       = cfg["risk"]["tp_rr"]

    if sig["direction"] == "bullish":
        entry = float(zone.bottom + off * (zone.top - zone.bottom))
        stop  = float(zone.bottom - pad_mult * cur_atr)
        tp1   = entry + rr * (entry - stop)
    else:
        entry = float(zone.top - off * (zone.top - zone.bottom))
        stop  = float(zone.top + pad_mult * cur_atr)
        tp1   = entry - rr * (stop - entry)

    return {"entry": entry, "stop": stop, "tp1": tp1, "atr": float(cur_atr)}

def score_signal(zone: Zone, sig: Dict, plan: Dict, cfg: dict, regime_tag: str, htf_bias: Optional[str]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    zs = min(1.0, max(0.0, zone.strength))
    zs_w = cfg["scoring"]["w_zone_strength"]
    score += 100 * zs * zs_w
    reasons.append(f"zone:{zs:.2f}")

    dist  = abs(plan["entry"] - sig["level"])
    clean = max(0.0, 1.0 - dist / max(plan["atr"], 1e-9))
    cl_w  = cfg["scoring"]["w_signal_clean"]
    score += 100 * clean * cl_w
    reasons.append(f"clean:{clean:.2f}")

    rg_w = cfg["scoring"]["w_regime"]
    if regime_tag == "trending" and sig["type"] == "BOS":
        score += 100 * 1.0 * rg_w; reasons.append("reg:trending✓")
    elif regime_tag == "ranging" and sig["type"] == "SFP":
        score += 100 * 1.0 * rg_w; reasons.append("reg:ranging✓")
    else:
        reasons.append(f"reg:{regime_tag}×")

    bias_w = cfg["scoring"]["w_bias"]
    if cfg["filters"]["use_htf_bias"] and htf_bias:
        ok = (htf_bias == sig["direction"])
        score += 100 * (1.0 if ok else 0.0) * bias_w
        reasons.append(f"bias:{htf_bias}{'✓' if ok else '×'}")

    return score, reasons

def dedupe_key(symbol: str, tf: str, sig: Dict) -> str:
    return f"{symbol}:{tf}:{sig['type']}:{sig['direction']}:{round(float(sig['level']), 2)}"

def post_discord(webhook: str, title: str, fields: List[Dict], footer: str, color: int = 0x5865F2):
    if not webhook:
        return
    payload = {
        "username": "StructureBot",
        "embeds": [{
            "title": title,
            "color": color,
            "fields": fields,
            "footer": {"text": footer}
        }]
    }
    try:
        with httpx.Client(timeout=15) as cli:
            cli.post(webhook, json=payload)
    except Exception as e:
        print(f"[DISCORD] post error: {e}")

# ---- zone persist helper ----
def save_zone(db: DB, symbol: str, tf: str, zone):
    if zone is None:
        return

    def zget(attr, default=None):
        if isinstance(zone, dict):
            return zone.get(attr, default)
        return getattr(zone, attr, default)

    kind   = zget("kind")
    top    = zget("top")
    bottom = zget("bottom")
    i_end  = zget("impulse_end_idx", zget("impulse_end", zget("end_idx")))
    streng = zget("strength", zget("score", 0.0))

    if kind is None or top is None or bottom is None:
        print(f"[ZONE] skip persist (missing fields) {symbol} {tf} kind={kind} top={top} bottom={bottom}")
        return

    try:
        db.upsert_zone(symbol, tf, kind, float(top), float(bottom),
                       int(i_end) if i_end is not None else 0,
                       float(streng) if streng is not None else 0.0)
    except Exception as e:
        print(f"[ZONE] persist error {symbol} {tf}: {e}")

def regime_tag_from_vol(ohlcv: np.ndarray, cfg: dict) -> str:
    h = ohlcv[:, 2]; l = ohlcv[:, 3]; c = ohlcv[:, 4]
    long_atr = atr(h, l, c, cfg["regime"]["atr_ma_len"])
    rng = np.mean(h[-50:] - l[-50:])
    return "trending" if (rng / max(long_atr, 1e-9)) >= cfg["regime"]["trend_ratio_min"] else "ranging"

def htf_bias_simple(ohlcv_htf: np.ndarray) -> Optional[str]:
    if ohlcv_htf.shape[0] < 20:
        return None
    closes = ohlcv_htf[:, 4]
    ma = np.mean(closes[-20:])
    return "bullish" if closes[-1] >= ma else "bearish"

def load_overrides(db: DB) -> Dict[Tuple[str, str], int]:
    try:
        if not db.enabled:
            return {}
        res = db.client.table("score_overrides").select("*").execute()
        rows = res.data or []
        return {(r["symbol"], r["timeframe"]): int(r["min_score_to_alert"]) for r in rows}
    except Exception:
        # likely the table doesn't exist yet
        return {}

def resolve_symbol(ex: ccxt.Exchange, sym: str) -> str:
    mkts = ex.load_markets()
    if sym in mkts:
        return sym
    if sym.endswith("/USDT") and f"{sym}:USDT" in mkts:
        return f"{sym}:USDT"
    if sym.endswith(":USDT") and sym.replace(":USDT", "") in mkts:
        return sym.replace(":USDT", "")
    return sym

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

    seen: deque[Tuple[str, datetime]] = deque(maxlen=2048)
    dedupe_minutes = int(cfg.get("dedupe_minutes", 30))

    overrides = load_overrides(db)
    last_ovr = time.time()

    # startup backfill
    if cfg.get("startup_backfill", {}).get("enabled", True):
        bf_bars = int(cfg["startup_backfill"]["lookback_bars"])
        max_per = int(cfg["startup_backfill"]["max_signals_per_market"])
        print(f"[BACKFILL] Starting (bars={bf_bars}, max_per_market={max_per})")

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
                if zone:
                    if cfg["debug"].get("log_zones", False):
                        save_zone(db, sym, tf, zone)
                    sig = check_bos_sfp(arr, zone, cfg)
                    if sig:
                        plan = build_plan(arr, zone, sig, cfg)
                        regime = regime_tag_from_vol(arr, cfg)
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
                        gate = overrides.get((sym, tf), gate_tf)
                        if score >= gate:
                            key = dedupe_key(sym, tf, sig)
                            now = datetime.now(timezone.utc)
                            # keep only recent keys
                            keep_after = now - timedelta(minutes=dedupe_minutes)
                            _tmp = [(k, t) for (k, t) in list(seen) if t >= keep_after]
                            seen.clear()
                            for kv in _tmp: seen.append(kv)
                            if not any(k == key for k, _ in seen):
                                seen.append((key, now))
                                title = f"RECENT {sig['type']} — {sym} {tf} ({sig['direction'].upper()})"
                                fields = [
                                    {"name":"Zone","value": f"{zone.kind} | Level: {sig['level']:.2f}", "inline":False},
                                    {"name":"Entry (limit)","value": f"{plan['entry']:.2f}", "inline":True},
                                    {"name":"Stop","value": f"{plan['stop']:.2f}", "inline":True},
                                    {"name":"TP1 (~{rr}R)".format(rr=cfg['risk']['tp_rr']),
                                     "value": f"{plan['tp1']:.2f}", "inline":True},
                                    {"name":"Zone Top / Bottom","value": f"{zone.top:.2f} / {zone.bottom:.2f}", "inline":True},
                                    {"name":"ATR","value": f"{plan['atr']:.2f}", "inline":True},
                                    {"name":"Score","value": f"{score:.1f} / 100", "inline":True},
                                ]
                                post_discord(webhook, title, fields,
                                             footer="BOS/SFP from last impulse wick→body zone • startup-backfill")
                                db.log_signal(sym, tf, sig, zone, plan, score, reasons, True, key)

    print(f"[INFO] Using timeframes: {tfs}")
    print(f"[INFO] Using symbols: {symbols}")

    last_hb = datetime.now(timezone.utc)
    HEART_MIN = int(cfg["debug"].get("heartbeat_minutes", 2))

    while True:
        for sym in symbols:
            for tf in tfs:
                print(f"[SCAN] {sym} {tf} …")
                try:
                    rsym = resolve_symbol(ex, sym)
                    ohlcv = ex.fetch_ohlcv(rsym, tf, limit=min(lb, 1500))
                except Exception as e:
                    print(f"[ERR] {sym} {tf}: {e}")
                    continue
                if not ohlcv or len(ohlcv) < 50:
                    print(f"[WAIT] {sym} {tf} — insufficient data")
                    continue

                arr  = np.array(ohlcv, dtype=float)
                zone = detect_impulse_and_zone(arr, cfg)
                if not zone:
                    print(f"[WAIT] {sym} {tf} — no valid impulse/zone yet")
                    continue

                if cfg["debug"].get("log_zones", False):
                    save_zone(db, sym, tf, zone)

                sig = check_bos_sfp(arr, zone, cfg)
                if not sig:
                    continue

                plan   = build_plan(arr, zone, sig, cfg)
                regime = regime_tag_from_vol(arr, cfg)
                bias   = None
                if cfg["filters"]["use_htf_bias"]:
                    htf_tf = cfg["filters"]["htf_timeframe"]
                    try:
                        htf = ex.fetch_ohlcv(rsym, htf_tf, limit=200)
                        bias = htf_bias_simple(np.array(htf, dtype=float))
                    except Exception:
                        bias = None

                score, reasons = score_signal(zone, sig, plan, cfg, regime, bias)

                gate_tf = cfg["scoring"]["tf_overrides"].get(tf, cfg["scoring"]["min_score_to_alert"])
                gate = overrides.get((sym, tf), gate_tf)
                if score < gate:
                    if cfg["debug"].get("log_scans", False):
                        print(f"[FILTER] {sym} {tf} {sig['type']} {sig['direction']} score {score:.1f} < gate {gate}")
                    continue

                # prune & check dedupe
                now = datetime.now(timezone.utc)
                keep_after = now - timedelta(minutes=dedupe_minutes)
                _tmp = [(k, t) for (k, t) in list(seen) if t >= keep_after]
                seen.clear()
                for kv in _tmp: seen.append(kv)
                key = dedupe_key(sym, tf, sig)
                if any(k == key for k, _ in seen):
                    if cfg["debug"].get("log_scans", False):
                        print(f"[DEDUPE] skip {key}")
                    continue
                seen.append((key, now))

                # alert
                rr_txt = cfg['risk']['tp_rr']
                title = f"{sig['type']} — {sym} {tf} ({sig['direction'].upper()})"
                fields = [
                    {"name":"Zone","value": f"{zone.kind} | Level: {sig['level']:.2f}", "inline":False},
                    {"name":"Close","value": f"{arr[-1,4]:.2f} | Time (ms): {int(arr[-1,0])}", "inline":False},
                    {"name":"Entry (limit)","value": f"{plan['entry']:.2f}", "inline":True},
                    {"name":"Stop","value": f"{plan['stop']:.2f}", "inline":True},
                    {"name":f"TP1 (~{rr_txt}R)","value": f"{plan['tp1']:.2f}", "inline":True},
                    {"name":"Zone Top / Bottom","value": f"{zone.top:.2f} / {zone.bottom:.2f}", "inline":True},
                    {"name":"ATR","value": f"{plan['atr']:.2f}", "inline":True},
                    {"name":"Score","value": f"{score:.1f} / 100", "inline":True},
                ]
                reasons_str = f"reg:{regime} • zone:{zone.strength:.2f} • clean:{abs(plan['entry']-sig['level'])/max(plan['atr'],1e-9):.2f}"
                post_discord(
                    webhook, title, fields,
                    footer=f"BOS/SFP from last impulse wick→body zone • {reasons_str}",
                    color=0x2ecc71 if sig["direction"] == "bullish" else 0xe74c3c
                )

                db.log_signal(sym, tf, sig, zone, plan, score, reasons, False, key)

        if time.time() - last_ovr > 600:
            overrides = load_overrides(db)
            last_ovr = time.time()
            print(f"[OVERRIDES] reloaded {len(overrides)} rows")

        # heartbeat
        print(f"[HEARTBEAT] {datetime.now(timezone.utc).isoformat()} — sleeping {poll}s")
        time.sleep(poll)
