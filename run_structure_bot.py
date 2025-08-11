import os
import time
import yaml
from structurebot.data_feed import DataFeed
from structurebot.structure_engine import StructureEngine
from structurebot.utils import to_candles
from structurebot.notifier import Notifier
from structurebot.state import State

CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")

def load_cfg():
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

def make_payload(symbol, timeframe, cfg, zone, sig, candles):
    bar = candles[sig['idx']]
    payload = {
        "username": "StructureBot v1",
        "embeds": [
            {
                "title": f"{sig['type']} — {symbol} {timeframe}",
                "description": (
                    f"Zone: **{zone.kind}** | Level: **{sig['level']:.2f}**\n"
                    f"Direction: **{sig['direction']}**\n"
                    f"Close: **{bar.c:.2f}** | Time (ms): **{bar.ts}**\n"
                ),
                "fields": [
                    {"name": "Zone Top / Bottom", "value": f"{zone.top:.2f} / {zone.bottom:.2f}", "inline": True},
                    {"name": "Impulse End Index", "value": str(zone.impulse_end_idx), "inline": True},
                ],
                "footer": {"text": "BOS/SFP from last impulse wick→body zone"}
            }
        ]
    }
    return payload

if __name__ == "__main__":
    cfg = load_cfg()
    feed = DataFeed(cfg['exchange'])
    engine = StructureEngine(cfg)
    notify = Notifier(cfg.get('discord_webhook_url', ''))
    state = State()

    while True:
        for symbol in cfg['symbols']:
            for tf in cfg['timeframes']:
                try:
                    ohlcv = feed.fetch_ohlcv(symbol, tf, limit=cfg['lookback_bars'])
                    candles = to_candles(ohlcv)

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
                            notify.post(make_payload(symbol, tf, cfg, zone, sig, candles))

                except Exception as e:
                    print(f"[ERR] {symbol} {tf}: {e}")
                    continue

        time.sleep(cfg['poll_seconds'])
