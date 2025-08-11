import os, time, yaml, datetime
from structurebot.data_feed import DataFeed
from structurebot.structure_engine import StructureEngine
from structurebot.utils import to_candles
from structurebot.notifier import Notifier
from structurebot.state import State
from structurebot.indicators import atr

CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")

def load_cfg():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def heartbeat(now, last, minutes):
    if minutes <= 0: return False, last
    if (now - last) >= minutes * 60:
        return True, now
    return False, last

def compute_trade_plan(cfg, candles, zone, sig):
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
            entry = level * (1 - rpct); stop = level - recent_atr * stop_mult
        else:
            side, level = "SHORT", zone.bottom
            entry = level * (1 + rpct); stop = level + recent_atr * stop_mult
    else:
        if "bearish" in sig["direction"]:
            side, level = "SHORT", zone.top
            entry = level * (1 + rpct); stop = level + recent_atr * stop_mult
        else:
            side, level = "LONG", zone.bottom
            entry = level * (1 - rpct); stop = level - recent_atr * stop_mult

    risk_per = abs(entry - stop)
    tp1 = entry + rr * risk_per if side == "LONG" else entry - rr * risk_per
    return {"side": side, "entry": float(entry), "stop": float(stop), "tp1": float(tp1), "atr": recent_atr}

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
        "embeds": [{
            "title": title,
            "description": desc,
            "fields": [
                {"name":"Entry (limit)","value":f"{plan['entry']:.2f}","inline":True},
                {"name":"Stop","value":f"{plan['stop']:.2f}","inline":True},
                {"name":f"TP1 (~{cfg['risk']['tp_rr']}R)","value":f"{plan['tp1']:.2f}","inline":True},
                {"name":"Zone Top / Bottom","value":f"{zone.top:.2f} / {zone.bottom:.2f}","inline":True},
                {"name":"ATR","value":f"{plan['atr']:.2f}","inline":True},
            ],
            "footer":{"text":"BOS/SFP from last impulse wick→body zone • Entries are retest-based"}
        }]
    }

if __name__ == "__main__":
    cfg = load_cfg()
    feed = DataFeed(cfg["exchange"])
    eng  = StructureEngine(cfg)

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or cfg.get("discord_webhook_url","")
    notify = Notifier(webhook_url)
    state  = State()

    # Debug flags
    dbg = cfg.get("debug", {})
    LOG_SCANS  = bool(dbg.get("log_scans", True))
    LOG_ZONES  = bool(dbg.get("log_zones", True))
    HEART_MIN  = int(dbg.get("heartbeat_minutes", 1))
    POST_DEBUG = bool(dbg.get("post_debug_to_discord", False))

    supported_tfs = set((feed.exchange.timeframes or {}).keys())
    req_tfs = cfg["timeframes"]
    valid_tfs = [tf for tf in req_tfs if not supported_tfs or tf in supported_tfs] or req_tfs
    print(f"[INFO] Using timeframes: {valid_tfs}")

    last_heartbeat = 0
    while True:
        loop_started = time.time()
        for symbol in cfg["symbols"]:
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
                        continue

                    if LOG_ZONES:
                        print(f"[ZONE] {symbol} {tf} — {zone.kind} {zone.bottom:.2f} → {zone.top:.2f}")

                    bos = eng.bos_signal(candles, zone)
                    sfp = eng.sfp_signal(candles, zone)

                    fired = False
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
                        fired = True

                    if not fired and POST_DEBUG:
                        # Optional: post a lightweight debug embed to Discord
                        notify.post({
                            "username": "StructureBot v1 (debug)",
                            "embeds": [{
                                "title": f"Watching {symbol} {tf}",
                                "description": f"{zone.kind} zone {zone.bottom:.2f} → {zone.top:.2f}\nNo BOS/SFP yet."
                            }]
                        })

                except Exception as e:
                    print(f"[ERR] {symbol} {tf} ({cfg['exchange']}): {e}")
                    continue

        now = time.time()
        hb, last_heartbeat = heartbeat(now, last_heartbeat, HEART_MIN)
        if hb:
            print(f"[HEARTBEAT] {datetime.datetime.utcnow().isoformat()}Z — cycle OK, sleeping {cfg['poll_seconds']}s")

        # keep loop timing simple
        time.sleep(cfg["poll_seconds"])
