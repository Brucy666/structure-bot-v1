import os
import time
from datetime import datetime, timezone, timedelta

import ccxt
import httpx
import yaml

from structurebot.db import DB


CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")
SIGNALS_TABLE = os.getenv("SB_SIGNALS_TABLE", "signals")  # configurable table name
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
TF_MS = {
    "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000,
    "30m": 1800000, "1h": 3600000, "4h": 14400000, "12h": 43200000, "1d": 86400000
}


def load_cfg() -> dict:
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def resolve_symbol(exchange: ccxt.Exchange, sym: str) -> str:
    markets = exchange.load_markets()
    if sym in markets:
        return sym
    if sym.endswith("/USDT") and f"{sym}:USDT" in markets:
        return f"{sym}:USDT"
    if sym.endswith(":USDT") and sym.replace(":USDT", "") in markets:
        return sym.replace(":USDT", "")
    return sym


def post_discord_outcome(s: dict, kind: str, price: float, rr: float, ts_iso: str) -> None:
    if not WEBHOOK:
        return
    title = f"{kind.upper()} — {s['symbol']} {s['timeframe']}"
    color = 0x2ecc71 if kind == "tp" else 0xe74c3c
    fields = [
        {"name": "Entry", "value": f"{float(s['entry']):.2f}", "inline": True},
        {"name": "Stop", "value": f"{float(s['stop']):.2f}", "inline": True},
        {"name": "TP1", "value": f"{float(s['tp1']):.2f}", "inline": True},
        {"name": "Exit", "value": f"{price:.2f}", "inline": True},
        {"name": "RR", "value": f"{rr:.2f}", "inline": True},
    ]
    payload = {
        "username": "StructureBot",
        "embeds": [{
            "title": title,
            "color": color,
            "description": f"{s['type']} • {s['direction']} • opened {s['created_at']}",
            "fields": fields,
            "footer": {"text": f"exited {ts_iso}"},
        }]
    }
    try:
        with httpx.Client(timeout=15) as x:
            x.post(WEBHOOK, json=payload)
    except Exception as e:  # noqa: BLE001
        print(f"[OUTCOME] discord post err: {e}")


if __name__ == "__main__":
    cfg = load_cfg()
    db = DB()

    ex = ccxt.bybit({"enableRateLimit": True})
    tf = cfg["outcomes"]["tf_for_check"]
    max_age_h = int(cfg["outcomes"]["max_age_hours"])
    miss_after_h = int(cfg["outcomes"]["miss_after_hours"])
    poll_s = int(cfg["outcomes"]["poll_seconds"])
    post_outcomes = bool(cfg["outcomes"].get("post_to_discord", False))

    print(f"[OUTCOME] start tf={tf}, max_age={max_age_h}h, miss_after={miss_after_h}h | table={SIGNALS_TABLE}")

    while True:
        try:
            since_iso = (datetime.now(timezone.utc) - timedelta(hours=max_age_h)).isoformat()
            rows = db.fetch_recent_signals(since_iso, limit=200)
        except Exception as e:  # noqa: BLE001
            print(f"[OUTCOME] supabase select err: {e}")
            rows = []

        for s in rows:
            # already decided?
            if s.get("outcome") in ("tp", "sl", "missed", "timeout"):
                continue

            symbol = resolve_symbol(ex, s["symbol"])
            side = "LONG" if float(s["entry"]) < float(s["tp1"]) else "SHORT"

            created_at = datetime.fromisoformat(
                s["created_at"].replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            age_h = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600

            # mark missed if never entered within miss_after window
            if not s.get("entered_at") and age_h >= miss_after_h:
                db.update_signal(s["id"], {
                    "outcome": "missed",
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                })
                print(f"[OUTCOME] {s['id']} missed")
                continue

            # fetch candles from creation onward
            since_ms = int(created_at.timestamp() * 1000) - TF_MS.get(tf, 60000)
            try:
                ohlcv = ex.fetch_ohlcv(symbol, tf, since=since_ms, limit=1500)
            except Exception as e:  # noqa: BLE001
                print(f"[OUTCOME] fetch err {symbol} {tf}: {e}")
                continue
            if not ohlcv:
                continue

            entry = float(s["entry"])
            stop = float(s["stop"])
            tp1 = float(s["tp1"])

            # detect/confirm entry
            entered_ms = None
            if s.get("entered_at"):
                entered_ms = int(datetime.fromisoformat(
                    s["entered_at"].replace("Z", "+00:00")
                ).timestamp() * 1000)
            else:
                for ts, o, h, l, c, v in ohlcv:
                    if l <= entry <= h:
                        entered_ms = ts
                        db.update_signal(s["id"], {
                            "entered_at": datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat(),
                            "outcome": "open",
                            "last_checked": datetime.now(timezone.utc).isoformat(),
                        })
                        print(f"[OUTCOME] {s['id']} entered")
                        break

                if not entered_ms:
                    db.update_signal(s["id"], {"last_checked": datetime.now(timezone.utc).isoformat()})
                    continue

            # after entry: first touch TP or SL wins
            # find first candle >= entry time
            start_idx = 0
            for i, row in enumerate(ohlcv):
                if row[0] >= entered_ms:
                    start_idx = i
                    break

            hit = None  # (kind, price, index)
            for i in range(start_idx, len(ohlcv)):
                ts, o, h, l, c, v = ohlcv[i]
                if side == "LONG":
                    if l <= stop:
                        hit = ("sl", stop, i)
                        break
                    if h >= tp1:
                        hit = ("tp", tp1, i)
                        break
                else:
                    if h >= stop:
                        hit = ("sl", stop, i)
                        break
                    if l <= tp1:
                        hit = ("tp", tp1, i)
                        break

            if hit:
                kind, price, i = hit
                ex_time = datetime.fromtimestamp(ohlcv[i][0] / 1000, timezone.utc).isoformat()
                risk = abs(entry - stop)
                rr = (price - entry) / risk if side == "LONG" else (entry - price) / risk

                db.update_signal(s["id"], {
                    "outcome": kind,
                    "exited_at": ex_time,
                    "exit_price": float(price),
                    "rr_achieved": float(rr),
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                })
                print(f"[OUTCOME] {s['id']} {symbol} -> {kind} at {price} (RR={rr:.2f})")

                if post_outcomes:
                    post_discord_outcome(s, kind, float(price), float(rr), ex_time)

            else:
                if age_h >= max_age_h:
                    db.update_signal(s["id"], {"outcome": "timeout", "last_checked": datetime.now(timezone.utc).isoformat()})
                    print(f"[OUTCOME] {s['id']} timeout")
                else:
                    db.update_signal(s["id"], {"outcome": "open", "last_checked": datetime.now(timezone.utc).isoformat()})

        print(f"[OUTCOME] heartbeat {datetime.now(timezone.utc).isoformat()} — sleeping {poll_s}s")
        time.sleep(poll_s)
