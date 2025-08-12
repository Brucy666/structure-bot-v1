import os, time
from typing import Optional, Dict, Any

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

class DB:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.enabled = bool(self.url and self.key and create_client)
        self.client: Optional[Client] = create_client(self.url, self.key) if self.enabled else None
        if not self.enabled:
            print("[DB] Supabase disabled (missing env or package)")

    def _try(self, fn):
        if not self.enabled: return
        last = None
        for _ in range(3):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(0.4)
        print(f"[DB] error: {last}")

    def upsert_zone(self, symbol, tf, zone):
        return self._try(lambda: self.client.table("zones").upsert({
            "symbol": symbol, "timeframe": tf, "kind": zone.kind,
            "top": float(zone.top), "bottom": float(zone.bottom),
            "impulse_end_idx": int(zone.impulse_end_idx),
            "strength": float(getattr(zone, "strength", 1.0))
        }, on_conflict="symbol,timeframe,kind,top,bottom").execute())

    def log_signal(self, symbol, tf, sig, zone, plan, score, reasons, is_backfill, dedupe_key):
        return self._try(lambda: self.client.table("signals").insert({
            "symbol": symbol, "timeframe": tf,
            "type": sig["type"], "direction": sig["direction"],
            "zone_kind": zone.kind, "zone_top": float(zone.top), "zone_bottom": float(zone.bottom),
            "level": float(sig["level"]), "idx": int(sig["idx"]),
            "entry": float(plan["entry"]), "stop": float(plan["stop"]), "tp1": float(plan["tp1"]),
            "atr": float(plan["atr"]), "score": float(score),
            "reasons": ",".join(reasons), "is_backfill": bool(is_backfill),
            "dedupe_key": dedupe_key
        }).execute())

    def update_signal(self, signal_id: int, fields: Dict[str, Any]):
        return self._try(lambda: self.client.table("signals").update(fields).eq("id", signal_id).execute())
