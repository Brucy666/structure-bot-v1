import os
import time
import yaml
from datetime import datetime, timezone

from structurebot.data_feed import DataFeed
from structurebot.structure_engine import StructureEngine, Zone, Candle
from structurebot.utils import to_candles
from structurebot.notifier import Notifier
from structurebot.state import State
from structurebot.indicators import atr

CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")


def load_cfg():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def heartbeat(now_ts, last_ts, minutes: int):
    if minutes <= 0:
        return False, last_ts
    if now_ts - last_ts >= minutes * 60:
        return True, now_ts
    return False, last_ts


def compute_trade_plan(cfg, candles, zone, sig):
    """Builds trade plan: entry, stop, tp1, side."""
    import numpy as np

    o = np.array([x.o for x in candles], float)
    h = np.array([x.h for x in candles], float)
    l = np.array([x.l for x in candles], float)
    c = np.array([x.c for x in candles], float)
    a = atr(o, h, l, c, length=cfg["impulse"]["atr_len"])
    recent_atr = float(a[-1]) if a[-1] == a[-1] else 0.0

    rpct = cfg["risk"]["retest_offset_pct"] / 100.0
    stop_mult = cfg["risk"]["stop_atr_mult"]
    rr = cfg["risk"]["tp_rr"]

    if sig["type"] == "BOS":
        if sig["direction"] == "bullish":
            side, level = "LONG", zone.top
            entry = level * (1 - rpct)
            stop = level - recent_atr * stop_mult
        else:
            side, level = "SHORT", zone.bottom
            entry = level * (1 + rpct)
            stop = level + recent_atr * stop_mult
    else:
        if "bearish" in sig["direction"]:
            side, level = "SHORT", zone.top
            entry = level * (1 + rpct)
            stop = level + recent_atr * stop_mult
        else:
            side, level = "LONG", zone.bottom
            entry = level * (1 - rpct)
            stop = level - recent_atr * stop_mult

    risk_per = abs(entry - stop)
    tp1 = entry + rr * risk_per if side == "LONG" else entry - rr * risk_per

    return {
        "side": side,
        "entry": float(entry),
        "stop": float(stop),
        "tp1": float(tp1),
        "atr": float(recent_atr),
    }


def make_payload(symbol, timeframe, cfg, zone, sig, candles, plan):
    bar = candles[sig["idx"]]
    title = f"{sig['type']} — {symbol} {timeframe} ({plan['side']})"
    desc = (
        f"Zone: **{zone.kind}** | Level: **{sig['level']:.2f}**\n"
        f"Direction: **{sig['direction']}**\n"
        f"Close: **{bar.c:.2f}** | Time (ms): **{bar.ts}**"
    )
    return {
        "username": "StructureBot v1",
        "embeds": [
            {
                "title": title,
                "description": desc,
                "fields": [
                    {"name": "Entry (limit)", "value": f"{plan['entry']:.2f}", "inline": True},
                    {"name": "Stop", "value": f"{plan['stop']:.2f}", "inline": True},
                    {"name": f"TP1 (~{cfg['risk']['tp_rr']}R)", "value": f"{plan['tp1']:.2f}", "inline": True},
                    {"name": "Zone Top / Bottom", "value": f"{zone.top:.2f} / {zone.bottom:.2f}", "inline": True},
                    {"name": "ATR", "value": f"{plan['atr']:.2f}", "inline": True},
                ],
                "footer": {"text": "BOS/SFP from last impulse wick→body zone • Entries are retest-based"},
            }
        ],
    }


if __name__ == "__main__":
    cfg = load_cfg()
    feed = DataFeed(cfg["exchange"])
    eng = StructureEngine(cfg)
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or cfg.get("discord_webhook_url", "")
    notify = Notifier(webhook_url)
    state = State()

    # -------- ONE-TIME TEST FIRE --------
    from time import sleep
    test_zone = Zone(kind="bearish", bottom=30000.0, top=30050.0, impulse_end_idx=100)
    test_sig = {"type": "BOS", "direction": "bullish", "level": 30050.0, "idx": 101}
    test_candles = [Candle(ts=0, o=0, h=0, l=0, c=0, v=0)] * 102
    test_candles[test_sig["idx"]] = Candle(ts=1723372840000, o=30020.0, h=30060.0, l=29980.0, c=30070.0, v=1)
    test_plan = {"side": "LONG", "entry": 30045.0, "stop": 29990.0, "tp1": 30130.0, "atr": 25.0}
    notify.post(make_payload("BTC/USDT", "15m", cfg, test_zone, test_sig, test_candles, test_plan))
    print("[TEST] Sent fake BOS LONG alert to Discord — continuing live scanning...")
    sleep(5)
    # ------------------------------------

    dbg = cfg.get("debug", {})
    LOG_SCANS = bool(dbg.get("log_scans", True))
    LOG_ZONES = bool(dbg.get("log_zones", True))
    HEART_MIN = int(dbg.get("heartbeat_minutes", 1))
    POST_DEBUG = bool(dbg.get("post_debug_to_discord", False))

    supported_tfs = set((feed.exchange.timeframes or {}).keys())
    req_tfs = cfg["timeframes"]
    valid_tfs = [tf for tf in req_tfs if not supported_tfs or tf in supported_tfs] or req_tfs
    print(f"[INFO] Using timeframes: {valid_tfs}")

    markets = feed.exchange.load_markets()

    def resolve_symbol(sym: str):
        if sym in markets:
            return sym
        if sym.endswith("/USDT"):
            alt = sym + ":USDT"
            if alt in markets:
                print(f"[SYMBOL] Mapped {sym} -> {alt}")
                return alt
        if sym.endswith(":USDT"):
            base = sym.replace(":USDT", "")
            if base in markets:
                print(f"[SYMBOL] Mapped {sym} -> {base}")
                return base
        print(f"[SYMBOL] UNSUPPORTED on {cfg['exchange']}: {sym}")
        return None

    resolved_symbols = [s for s in (resolve_symbol(s) for s in cfg["symbols"]) if s]
    if not resolved_symbols:
        print("[WARN] No requested symbols are tradable.")
    else:
        print(f"[INFO] Using symbols: {resolved_symbols}")

    last_heartbeat = 0

    while True:
        for symbol in resolved_symbols:
            for tf in valid_tfs:
                try:
                    if LOG_SCANS:
                        print(f"[SCAN] {symbol} {tf} …")

                    ohlcv = feed.fetch_ohlcv(symbol, tf, limit=cfg["lookback_bars"])
                    candles = to_candles(ohlcv)
                    if len(candles) < 20:
                        if LOG_SCANS:
                            print(f"[SKIP] {symbol} {tf} — not enough candles")
                        continue

                    impulse = eng.detect_last_impulse(candles)
                    zone = eng.make_zone_from_impulse(candles, impulse)
                    if not zone:
                        if LOG_SCANS:
                            print(f"[WAIT] {symbol} {tf} — no valid impulse/zone yet")
                        if POST_DEBUG:
                            notify.post({"username": "StructureBot v1 (debug)",
                                         "embeds": [{"title": f"Watching {symbol} {tf}", "description": "No valid impulse/zone yet."}]})
                        continue

                    if LOG_ZONES:
                        print(f"[ZONE] {symbol} {tf} — {zone.kind} {zone.bottom:.2f} → {zone.top:.2f}")

                    bos = eng.bos_signal(candles, zone)
                    sfp = eng.sfp_signal(candles, zone)

                    for sig in [bos, sfp]:
                        if not sig:
                            continue
                        key = f"{symbol}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],2)}"
                        if not state.can_alert(key, cfg["dedupe_minutes"]):
                            print(f"[DEDUPE] {key} — throttled")
                            continue

                        plan = compute_trade_plan(cfg, candles, zone, sig)
                        notify.post(make_payload(symbol, tf, cfg, zone, sig, candles, plan))
                        print(f"[ALERT] {symbol} {tf} — {sig['type']} {sig['direction']} @ {sig['level']:.2f} → {plan['side']}")

                    if not bos and not sfp and POST_DEBUG:
                        notify.post({"username": "StructureBot v1 (debug)",
                                     "embeds": [{"title": f"Watching {symbol} {tf}",
                                                 "description": f"{zone.kind} zone {zone.bottom:.2f} → {zone.top:.2f}\nNo BOS/SFP yet."}]})

                except Exception as e:
                    print(f"[ERR] {symbol} {tf}: {e}")
                    continue

        now = time.time()
        hb, last_heartbeat = heartbeat(now, last_heartbeat, HEART_MIN)
        if hb:
            print(f"[HEARTBEAT] {datetime.now(timezone.utc).isoformat()} — cycle OK, sleeping {cfg['poll_seconds']}s")

        time.sleep(cfg["poll_seconds"])
