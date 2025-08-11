import os
import time
import yaml
from structurebot.data_feed import DataFeed
from structurebot.structure_engine import StructureEngine
from structurebot.utils import to_candles
from structurebot.notifier import Notifier
from structurebot.state import State
from structurebot.indicators import atr

CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")

def load_cfg():
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

def compute_trade_plan(symbol, timeframe, cfg, candles, zone, sig):
    """
    Returns dict with side (LONG/SHORT), entry, stop, tp1.
    Logic:
    - BOS bullish → LONG on retest of broken resistance (zone.top) with tiny offset.
    - BOS bearish → SHORT on retest of broken support (zone.bottom) with tiny offset.
    - SFP bearish-failed-break → SHORT from zone.top.
    - SFP bullish-failed-break → LONG from zone.bottom.
    Stops use recent ATR; TP1 by RR.
    """
    last_close = candles[sig['idx']].c
    # ATR on closes for robustness
    o = [x.o for x in candles]
    h = [x.h for x in candles]
    l = [x.l for x in candles]
    c = [x.c for x in candles]
    a = atr(
        __import__("numpy").array(o, float),
        __import__("numpy").array(h, float),
        __import__("numpy").array(l, float),
        __import__("numpy").array(c, float),
        length=cfg['impulse']['atr_len']
    )
    recent_atr = float(a[-1]) if a[-1] == a[-1] else 0.0  # handle NaN

    rpct = cfg['risk']['retest_offset_pct'] / 100.0
    stop_mult = cfg['risk']['stop_atr_mult']
    rr = cfg['risk']['tp_rr']

    if sig['type'] == 'BOS':
        if sig['direction'] == 'bullish':
            side = 'LONG'
            level = zone.top
            entry = level * (1 - rpct)
            stop = level - recent_atr * stop_mult
        else:
            side = 'SHORT'
            level = zone.bottom
            entry = level * (1 + rpct)
            stop = level + recent_atr * stop_mult
    else:  # SFP
        if 'bearish' in sig['direction']:
            side = 'SHORT'
            level = zone.top
            entry = level * (1 + rpct)
            stop = level + recent_atr * stop_mult
        else:
            side = 'LONG'
            level = zone.bottom
            entry = level * (1 - rpct)
            stop = level - recent_atr * stop_mult

    # risk per unit
    risk_per = abs(entry - stop)
    if risk_per == 0:
        tp1 = entry
    else:
        if side == 'LONG':
            tp1 = entry + rr * risk_per
        else:
            tp1 = entry - rr * risk_per

    return {
        "side": side,
        "entry": float(entry),
        "stop": float(stop),
        "tp1": float(tp1),
        "atr": recent_atr
    }

def make_payload(symbol, timeframe, cfg, zone, sig, candles, plan):
    bar = candles[sig['idx']]
    title = f"{sig['type']} — {symbol} {timeframe} ({plan['side']})"
    desc_lines = [
        f"Zone: **{zone.kind}** | Level: **{sig['level']:.2f}**",
        f"Direction: **{sig['direction']}**",
        f"Close: **{bar.c:.2f}** | Time (ms): **{bar.ts}**",
    ]
    payload = {
        "username": "StructureBot v1",
        "embeds": [
            {
                "title": title,
                "description": "\n".join(desc_lines),
                "fields": [
                    {"name": "Entry (limit)", "value": f"{plan['entry']:.2f}", "inline": True},
                    {"name": "Stop", "value": f"{plan['stop']:.2f}", "inline": True},
                    {"name": "TP1 (~{cfg[risk][tp_rr]}R)".replace("{cfg[risk][tp_rr]}", str(cfg['risk']['tp_rr'])), "value": f"{plan['tp1']:.2f}", "inline": True},
                    {"name": "Zone Top / Bottom", "value": f"{zone.top:.2f} / {zone.bottom:.2f}", "inline": True},
                    {"name": "ATR", "value": f"{plan['atr']:.2f}", "inline": True},
                ],
                "footer": {"text": "BOS/SFP from last impulse wick→body zone • Entries are retest-based"}
            }
        ]
    }
    return payload

if __name__ == "__main__":
    cfg = load_cfg()
    feed = DataFeed(cfg['exchange'])
    engine = StructureEngine(cfg)

    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL') or cfg.get('discord_webhook_url', '')
    notify = Notifier(webhook_url)
    state = State()

    supported_tfs = set((feed.exchange.timeframes or {}).keys())
    requested_tfs = cfg['timeframes']
    if supported_tfs:
        valid_tfs = [tf for tf in requested_tfs if tf in supported_tfs] or requested_tfs
    else:
        valid_tfs = requested_tfs

    print(f"[INFO] Using timeframes: {valid_tfs}")

    while True:
        for symbol in cfg['symbols']:
            for tf in valid_tfs:
                try:
                    ohlcv = feed.fetch_ohlcv(symbol, tf, limit=cfg['lookback_bars'])
                    candles = to_candles(ohlcv)
                    if len(candles) < 20:
                        continue

                    impulse = engine.detect_last_impulse(candles)
                    zone = engine.make_zone_from_impulse(candles, impulse)
                    if not zone:
                        continue

                    bos = engine.bos_signal(candles, zone)
                    sfp = engine.sfp_signal(candles, zone)

                    for sig in [bos, sfp]:
                        if not sig:
                            continue
                        key = f"{symbol}:{tf}:{sig['type']}:{sig['direction']}:{round(sig['level'],2)}"
                        if state.can_alert(key, cfg['dedupe_minutes']):
                            plan = compute_trade_plan(symbol, tf, cfg, candles, zone, sig)
                            notify.post(make_payload(symbol, tf, cfg, zone, sig, candles, plan))

                except Exception as e:
                    print(f"[ERR] {symbol} {tf} ({cfg['exchange']}): {e}")
                    continue

        time.sleep(cfg['poll_seconds'])
