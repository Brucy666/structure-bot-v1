import os, time, yaml
from datetime import datetime, timezone

from structurebot.data_feed import DataFeed
from structurebot.structure_engine import StructureEngine
from structurebot.utils import to_candles
from structurebot.notifier import Notifier
from structurebot.state import State
from structurebot.indicators import atr
from structurebot.db import DB

CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")

def load_cfg():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def heartbeat(now_ts, last_ts, minutes: int):
    return (now_ts - last_ts >= minutes * 60, now_ts if now_ts - last_ts >= minutes * 60 else last_ts)

def in_session_utc(now_ts: float, sessions):
    from datetime import datetime, timezone
    t = datetime.fromtimestamp(now_ts, timezone.utc).time()
    cur = t.hour * 60 + t.minute
    for window in sessions:
        a, b = window.split("-")
        ah, am = map(int, a.split(":")); bh, bm = map(int, b.split(":"))
        if ah * 60 + am <= cur <= bh * 60 + bm:
            return True
    return False

def compute_regime(o, h, l, c, cfg):
    import numpy as np
    a = atr(o, h, l, c, length=cfg["impulse"]["atr_len"])
    if len(a) < cfg["regime"]["atr_ma_len"]:
        return "ranging"
    ma = np.nanmean(a[-cfg["regime"]["atr_ma_len"]:])
    ratio = (a[-1] / (ma if ma > 0 else 1e-9))
    return "trending" if ratio >= cfg["regime"]["trend_ratio_min"] else "ranging"

def compute_bias_htf(feed, symbol, tf, bars=200):
    try:
        ohlcv = feed.fetch_ohlcv(symbol, tf, limit=bars)
        closes = [x[4] for x in ohlcv]; highs = [x[2] for x in ohlcv]; lows = [x[3] for x in ohlcv]
        swing_hi = max(highs[:-1]) if len(highs) > 1 else highs[-1]
        swing_lo = min(lows[:-1]) if len(lows) > 1 else lows[-1]
        if closes[-1] > swing_hi: return "bullish"
        if closes[-1] < swing_lo: return "bearish"
        return "neutral"
    except Exception:
        return "neutral"

def compute_trade_plan(cfg, candles, zone, sig):
    import numpy as np
    o = np.array([x.o for x in candles], float); h = np.array([x.h for x in candles], float)
    l = np.array([x.l for x in candles], float); c = np.array([x.c for x in candles], float)
    a = atr(o, h, l, c, length=cfg["impulse"]["atr_len"])
    recent_atr = float(a[-1]) if a[-1] == a[-1] else 0.0

    rpct = cfg["risk"]["retest_offset_pct"] / 100.0
    stop_mult = cfg["risk"]["stop_atr_mult"]; rr = cfg["risk"]["tp_rr"]

    if sig["type"] == "BOS":
        if sig["direction"] == "bullish":
            side, level = "LONG", zone.top; entry = level * (1 - rpct); stop = level - recent_atr * stop_mult
        else:
            side, level = "SHORT", zone.bottom; entry = level * (1 + rpct); stop = level + recent_atr * stop_mult
    else:
        if "bearish" in sig["direction"]:
            side, level = "SHORT", zone.top; entry = level * (1 + rpct); stop = level + recent_atr * stop_mult
        else:
            side, level = "LONG", zone.bottom; entry = level * (1 - rpct); stop = level - recent_atr * stop_mult

    risk = abs(entry - stop) if entry is not None and stop is not None else 0.0
    tp1 = entry + rr * risk if side == "LONG" else entry - rr * risk
    return {"side": side, "entry": float(entry), "stop": float(stop), "tp1": float(tp1), "atr": float(recent_atr)}

def make_payload(symbol, timeframe, cfg, zone, sig, candles, plan, score, reasons):
    bar = candles[sig["idx"]]
    return {
        "username": "StructureBot",
        "embeds": [{
            "title": f"{sig['type']} — {symbol} {timeframe} ({plan['side']})",
            "description": f"Zone: **{zone.kind}** | Level: **{sig['level']:.2f}**\n"
                           f"Direction: **{sig['direction']}**\n"
                           f"Close: **{bar.c:.2f}** | Time (ms): **{bar.ts}**",
            "fields": [
                {"name": "Entry (limit)", "value": f"{plan['entry']:.2f}", "inline": True},
                {"name": "Stop", "value": f"{plan['stop']:.2f}", "inline": True},
                {"name": f"TP1 (~{cfg['risk']['tp_rr']}R)", "value": f"{plan['tp1']:.2f}", "inline": True},
                {"name": "Zone Top / Bottom", "value": f"{zone.top:.2f} / {zone.bottom:.2f}", "inline": True},
                {"name": "ATR", "value": f"{plan['atr']:.2f}", "inline": True},
                {"name": "Score", "value": f"{score:.1f} / 100", "inline": True},
            ],
            "footer": {"text": "BOS/SFP from last impulse wick→body zone • " + ", ".join(reasons)}
        }]
    }

def find_recent_signal(engine, candles, zone, cfg):
    if not zone or not candles: return None
    closes = [x.c for x in candles]; highs = [x.h for x in candles]; lows = [x.l for x in candles]
    last_bos = None
    if zone.kind == "bearish":
        for i in range(zone.impulse_end_idx + 1, len(candles)):
            if closes[i] > zone.top: last_bos = {"type":"BOS","direction":"bullish","level":zone.top,"idx":i}
    else:
        for i in range(zone.impulse_end_idx + 1, len(candles)):
            if closes[i] < zone.bottom: last_bos = {"type":"BOS","direction":"bearish","level":zone.bottom,"idx":i}
    last_sfp = None
    w = cfg["signals"]["sfp_window"]; min_pen = cfg["sfp_quality"]["min_penetration_pct"]; max_inside = cfg["sfp_quality"]["max_close_inside_pct"]
    if zone.kind == "bearish":
        for i in range(zone.impulse_end_idx + 1, len(candles)):
            if highs[i] > zone.top and closes[i] <= zone.top:
                pen = (highs[i] - zone.top) / zone.top; inside = abs(closes[i] - zone.top) / zone.top
                if pen >= min_pen and inside <= max_inside: last_sfp = {'type':'SFP','direction':'bearish-failed-break','level':zone.top,'idx':i}
            for j in range(i+1, min(i+1+w, len(candles))):
                if highs[i] > zone.top and closes[j] <= zone.top:
                    pen = (highs[i] - zone.top) / zone.top; inside = abs(closes[j] - zone.top) / zone.top
                    if pen >= min_pen and inside <= max_inside: last_sfp = {'type':'SFP','direction':'bearish-failed-break','level':zone.top,'idx':j}
    else:
        for i in range(zone.impulse_end_idx + 1, len(candles)):
            if lows[i] < zone.bottom and closes[i] >= zone.bottom:
                pen = (zone.bottom - lows[i]) / zone.bottom; inside = abs(closes[i] - zone.bottom) / zone.bottom
                if pen >= min_pen and inside <= max_inside: last_sfp = {'type':'SFP','direction':'bullish-failed-break','level':zone.bottom,'idx':i}
            for j in range(i+1, min(i+1+w, len(candles))):
                if lows[i] < zone.bottom and closes[j] >= zone.bottom:
                    pen = (zone.bottom - lows[i]) / zone.bottom; inside = abs(closes[j] - zone.bottom) / zone.bottom
                    if pen >= min_pen and inside <= max_inside: last_sfp = {'type':'SFP','direction':'bullish-failed-break','level':zone.bottom,'idx':j}
    if last_bos and last_sfp:
        return last_bos if last_bos["idx"] > last_sfp["idx"] else last_sfp
    return last_bos or last_sfp

if __name__ == "__main__":
    cfg = load_cfg()
    feed = DataFeed(cfg["exchange"])
    eng  = StructureEngine(cfg)
    notify = Notifier(os.environ.get("DISCORD_WEBHOOK_URL") or cfg.get("discord_webhook_url",""))
    state  = State()
    db     = DB()

    dbg = cfg.get("debug", {})
    LOG_SCANS  = bool(dbg.get("log_scans", True))
    LOG_ZONES  = bool(dbg.get("log_zones", False))
    HEART_MIN  = int(dbg.get("heartbeat_minutes", 2))

    supported_tfs = set((feed.exchange.timeframes or {}).keys())
    valid_tfs = [tf for tf in cfg["timeframes"] if not supported_tfs or tf in supported_tfs] or cfg["timeframes"]
    print(f"[INFO] Using timeframes: {valid_tfs}")

    markets = feed.exchange.load_markets()
    def resolve_symbol(sym: str):
        if sym in markets: return sym
        if sym.endswith("/USDT") and sym + ":USDT" in markets: return sym + ":USDT"
        if sym.endswith(":USDT") and sym.replace(":USDT","") in markets: return sym.replace(":USDT","")
        print(f"[SYMBOL] UNSUPPORTED on {cfg['exchange']}: {sym}"); return None
    symbols = [s for s in (resolve_symbol(s) for s in cfg["symbols"]) if s]
    print(f"[INFO] Using symbols: {symbols}")

    # -------- startup backfill (real impulses) + dedupe ----------
    if cfg.get("startup_backfill", {}).get("enabled", False):
        bf_limit = int(cfg["startup_backfill"]["max_signals_per_market"])
        bf_bars  = int(cfg["startup_backfill"]["lookback_bars"])
        print(f"[BACKFILL] Starting (bars={bf_bars}, max_per_market={bf_limit})")
        for sym in symbols:
            for tf in valid_tfs:
                posted = 0
                try:
                    candles = to_candles(feed.fetch_ohlcv(sym, tf, limit=bf_bars))
                    if len(candles) < 50: 
                        print(f"[BACKFILL] {sym} {tf} — not enough candles"); 
                        continue
                    impulse = eng.detect_last_impulse(candles)
                    zone = eng.make_zone_from_impulse(candles, impulse)
                    if not zone: 
                        print(f"[BACKFILL] {sym} {tf} — no zone"); 
                        continue
                    if LOG_ZONES: db.upsert_zone(sym, tf, zone)
                    sig = find_recent_signal(eng, candles, zone, cfg)
                    if not sig:
                        print(f"[BACKFILL] {sym} {tf} — no recent BOS/SFP"); 
                        continue
                    plan = compute_trade_plan(cfg, candles, zone, sig)
                    payload = make_payload(sym, tf, cfg, zone, sig, candles, plan, 99.0, ["startup-backfill"])
                    payload["embeds"][0]["title"] = "RECENT " + payload["embeds"][0]["title"]
                    key = f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],2)}"
                    if not state.can_alert(key, cfg["dedupe_minutes"]):
                        print(f"[BACKFILL] deduped {sym} {tf}")
                    else:
                        notify.post(payload)
                        db.log_signal(sym, tf, sig, zone, plan, 99.0, ["startup-backfill"], True, key)
                        print(f"[BACKFILL] Posted RECENT {sym} {tf}")
                        posted += 1
                        if posted >= bf_limit: continue
                except Exception as e:
                    print(f"[BACKFILL ERR] {sym} {tf}: {e}")
        print("[BACKFILL] Done")

    last_hb = 0
    while True:
        for sym in symbols:
            for tf in valid_tfs:
                try:
                    if LOG_SCANS: print(f"[SCAN] {sym} {tf} …")
                    candles = to_candles(feed.fetch_ohlcv(sym, tf, limit=cfg["lookback_bars"]))
                    if len(candles) < 20: 
                        if LOG_SCANS: print(f"[SKIP] {sym} {tf} — not enough candles"); 
                        continue
                    if cfg["filters"]["session_filter"] and not in_session_utc(time.time(), cfg["filters"]["sessions_utc"]):
                        if LOG_SCANS: print(f"[SKIP] {sym} {tf} — out of session"); 
                        continue

                    import numpy as np
                    o = np.array([x.o for x in candles], float); h = np.array([x.h for x in candles], float)
                    l = np.array([x.l for x in candles], float); c = np.array([x.c for x in candles], float)
                    reg = compute_regime(o, h, l, c, cfg)
                    if reg == "trending":
                        eng.cfg["impulse"]["body_min"] = cfg["regime"]["adapt_body_min"][1]; eng.cfg["impulse"]["atr_mult"] = cfg["regime"]["adapt_atr_mult"][1]
                    else:
                        eng.cfg["impulse"]["body_min"] = cfg["regime"]["adapt_body_min"][0]; eng.cfg["impulse"]["atr_mult"] = cfg["regime"]["adapt_atr_mult"][0]

                    bias = "neutral"
                    if cfg["filters"]["use_htf_bias"]:
                        bias = compute_bias_htf(feed, sym, cfg["filters"]["htf_timeframe"])

                    impulse = eng.detect_last_impulse(candles)
                    zone = eng.make_zone_from_impulse(candles, impulse)
                    if not zone: 
                        if LOG_SCANS: print(f"[WAIT] {sym} {tf} — no valid impulse/zone yet"); 
                        continue

                    if cfg["debug"]["log_zones"]: db.upsert_zone(sym, tf, zone)

                    bos = eng.bos_signal(candles, zone)
                    sfp = eng.sfp_signal(candles, zone)

                    for sig in (bos, sfp):
                        if not sig: continue
                        plan = compute_trade_plan(cfg, candles, zone, sig)

                        score = 0.0; reasons = []
                        if bias != "neutral":
                            ok = (bias == "bullish" and sig["direction"].startswith("bullish")) or (bias == "bearish" and sig["direction"].startswith("bearish"))
                            score += cfg["scoring"]["w_bias"] * (100 if ok else 0); reasons.append(f"bias:{bias}{'✓' if ok else '×'}")
                        reg_ok = (reg == "trending" and sig["type"] == "BOS") or (reg == "ranging" and sig["type"] == "SFP")
                        score += cfg["scoring"]["w_regime"] * (100 if reg_ok else 0); reasons.append(f"reg:{reg}{'✓' if reg_ok else '×'}")
                        zs_norm = max(min(getattr(zone, "strength", 1.0) / 5.0, 1.0), 0.0)
                        score += cfg["scoring"]["w_zone_strength"] * (zs_norm * 100); reasons.append(f"zone:{zs_norm:.2f}")
                        if sig["type"] == "BOS":
                            last_close = candles[sig["idx"]].c; clean = max(min(abs(last_close - sig["level"]) / max(1e-9, plan["atr"]), 1.0), 0.0)
                        else:
                            clean = 1.0
                        score += cfg["scoring"]["w_signal_clean"] * (clean * 100); reasons.append(f"clean:{clean:.2f}")
                        final_score = round(score, 1)

                        gate = cfg["scoring"]["tf_overrides"].get(tf, cfg["scoring"]["min_score_to_alert"])
                        if final_score < gate:
                            if LOG_SCANS: print(f"[FILTER] {sym} {tf} {sig['type']} score {final_score} < {gate}")
                            continue

                        key = f"{sym}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],2)}"
                        if not state.can_alert(key, cfg["dedupe_minutes"]):
                            if LOG_SCANS: print(f"[DEDUPE] {key} — throttled")
                            continue

                        payload = make_payload(sym, tf, cfg, zone, sig, candles, plan, final_score, reasons)
                        Notifier(os.environ.get("DISCORD_WEBHOOK_URL") or cfg.get("discord_webhook_url","")).post(payload)
                        db.log_signal(sym, tf, sig, zone, plan, final_score, reasons, False, key)
                        print(f"[ALERT] {sym} {tf} — {sig['type']} {sig['direction']} → {plan['side']} | score {final_score}")

                except Exception as e:
                    print(f"[ERR] {sym} {tf} ({cfg['exchange']}): {e}")

        now = time.time()
        ping, last_hb = heartbeat(now, last_hb, HEART_MIN=cfg["debug"]["heartbeat_minutes"])
        if ping:
            print(f"[HEARTBEAT] {datetime.now(timezone.utc).isoformat()} — OK, sleeping {cfg['poll_seconds']}s")
        time.sleep(cfg["poll_seconds"])
