import os, time, yaml
from datetime import datetime, timezone, timedelta
import ccxt

from structurebot.db import DB

CONFIG_FILE = os.environ.get("STRUCTURE_CONFIG", "config.yml")
TF_MS = {"1m":60000,"3m":180000,"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"4h":14400000,"12h":43200000,"1d":86400000}

def load_cfg():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def resolve_symbol(exchange, sym):
    markets = exchange.load_markets()
    if sym in markets: return sym
    if sym.endswith("/USDT") and sym + ":USDT" in markets: return sym + ":USDT"
    if sym.endswith(":USDT") and sym.replace(":USDT","") in markets: return sym.replace(":USDT","")
    return sym

if __name__ == "__main__":
    cfg = load_cfg()
    db  = DB()
    ex  = ccxt.bybit({"enableRateLimit": True})
    tf   = cfg["outcomes"]["tf_for_check"]
    max_age_h = int(cfg["outcomes"]["max_age_hours"])
    miss_after_h = int(cfg["outcomes"]["miss_after_hours"])
    poll_s = int(cfg["outcomes"]["poll_seconds"])

    print(f"[OUTCOME] start tf={tf}, max_age={max_age_h}h, miss_after={miss_after_h}h")

    while True:
        try:
            res = db.client.table("signals") \
                .select("*") \
                .gte("created_at", (datetime.now(timezone.utc) - timedelta(hours=max_age_h)).isoformat()) \
                .order("created_at", desc=True).limit(200).execute()
            rows = res.data or []
        except Exception as e:
            print(f"[OUTCOME] supabase select err: {e}")
            rows = []

        for s in rows:
            if s.get("outcome") in ("tp","sl","missed","timeout"): continue
            symbol = resolve_symbol(ex, s["symbol"])
            side   = "LONG" if s["entry"] < s["tp1"] else "SHORT"
            created_at = datetime.fromisoformat(s["created_at"].replace("Z","+00:00")).astimezone(timezone.utc)
            age_h = (datetime.now(timezone.utc) - created_at).total_seconds()/3600

            if not s.get("entered_at") and age_h >= miss_after_h:
                db.update_signal(s["id"], {"outcome":"missed","last_checked":datetime.now(timezone.utc).isoformat()})
                print(f"[OUTCOME] {s['id']} missed"); 
                continue

            since_ms = int(created_at.timestamp()*1000) - TF_MS.get(tf,60000)
            try:
                ohlcv = ex.fetch_ohlcv(symbol, tf, since=since_ms, limit=1500)
            except Exception as e:
                print(f"[OUTCOME] fetch err {symbol} {tf}: {e}"); 
                continue
            if not ohlcv: continue

            entry = float(s["entry"]); stop = float(s["stop"]); tp1 = float(s["tp1"])

            # find or confirm entry
            entered_ms = None
            if s.get("entered_at"):
                entered_ms = int(datetime.fromisoformat(s["entered_at"].replace("Z","+00:00")).timestamp()*1000)
            else:
                for ts, o,h,l,c,v in ohlcv:
                    if l <= entry <= h:
                        entered_ms = ts; 
                        db.update_signal(s["id"], {"entered_at": datetime.fromtimestamp(ts/1000, timezone.utc).isoformat(),
                                                    "outcome":"open","last_checked":datetime.now(timezone.utc).isoformat()})
                        print(f"[OUTCOME] {s['id']} entered")
                        break
                if not entered_ms:
                    db.update_signal(s["id"], {"last_checked":datetime.now(timezone.utc).isoformat()})
                    continue

            # after entry, check TP/SL first touch
            after = [row for row in ohlcv if row[0] >= entered_ms]
            hit = None
            for ts, o,h,l,c,v in after:
                if side=="LONG":
                    if l <= stop: hit=("sl", stop, ts); break
                    if h >= tp1: hit=("tp", tp1, ts); break
                else:
                    if h >= stop: hit=("sl", stop, ts); break
                    if l <= tp1: hit=("tp", tp1, ts); break

            if hit:
                kind, price, ts = hit
                risk = abs(entry - stop)
                rr = (price - entry)/risk if side=="LONG" else (entry - price)/risk
                db.update_signal(s["id"], {
                    "outcome": kind, "exited_at": datetime.fromtimestamp(ts/1000, timezone.utc).isoformat(),
                    "exit_price": float(price), "rr_achieved": float(rr), "last_checked": datetime.now(timezone.utc).isoformat()
                })
                print(f"[OUTCOME] {s['id']} -> {kind} RR={rr:.2f}")
            else:
                if age_h >= max_age_h:
                    db.update_signal(s["id"], {"outcome":"timeout","last_checked":datetime.now(timezone.utc).isoformat()})
                    print(f"[OUTCOME] {s['id']} timeout")
                else:
                    db.update_signal(s["id"], {"outcome":"open","last_checked":datetime.now(timezone.utc).isoformat()})

        print(f"[OUTCOME] heartbeat {datetime.now(timezone.utc).isoformat()} — sleeping {poll_s}s")
        time.sleep(poll_s)
