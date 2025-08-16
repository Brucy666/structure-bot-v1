import os
import time
from typing import Any, Callable, Dict, Optional

from supabase import create_client, Client


class DB:
    """
    Thin Supabase wrapper with:
      - env-driven table name for StructureBot signals
      - simple retry helper
      - convenience methods the bot/outcome worker use
    """

    def __init__(self) -> None:
        self.url: str = os.getenv("SUPABASE_URL", "")
        self.key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.signals_table: str = os.getenv("SB_SIGNALS_TABLE", "signals")  # <— configurable
        self.enabled: bool = bool(self.url and self.key)

        self.client: Optional[Client] = None
        if self.enabled:
            self.client = create_client(self.url, self.key)
            print(f"[DB] Connected to {self.url} | signals_table={self.signals_table}")
        else:
            print("[DB] Supabase disabled (missing URL or KEY)")

    # ----- internals -----
    def _try(self, fn: Callable[[], Any], tag: str = "db", tries: int = 3, delay: float = 0.6) -> Any:
        last_exc = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last_exc = e
                print(f"[DB] {tag} error (attempt {i+1}/{tries}): {e}")
                time.sleep(delay)
        raise last_exc  # surface after retries

    # ----- zones -----
    def upsert_zone(self, symbol: str, timeframe: str, kind: str,
                    top: float, bottom: float, impulse_end_idx: int, strength: float) -> None:
        if not self.enabled or not self.client:
            return
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "kind": kind,
            "top": float(top),
            "bottom": float(bottom),
            "impulse_end_idx": int(impulse_end_idx),
            "strength": float(strength),
        }
        self._try(
            lambda: self.client.table("zones")
            .upsert(payload, on_conflict="symbol,timeframe,kind,top,bottom")
            .execute(),
            tag="zones.upsert",
        )

    # ----- signals (insert + update) -----
    def log_signal(
        self,
        symbol: str,
        timeframe: str,
        sig: Dict[str, Any],      # expects keys: type, direction, level, idx
        zone: Any,                # object with .kind .top .bottom
        plan: Dict[str, float],   # expects keys: entry, stop, tp1, atr
        score: float,
        reasons: Any,
        is_backfill: bool,
        dedupe_key: str,
    ) -> None:
        if not self.enabled or not self.client:
            return

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "type": sig["type"],
            "direction": sig["direction"],
            "zone_kind": getattr(zone, "kind", zone["kind"] if isinstance(zone, dict) else None),
            "zone_top": float(getattr(zone, "top", zone["top"] if isinstance(zone, dict) else 0.0)),
            "zone_bottom": float(getattr(zone, "bottom", zone["bottom"] if isinstance(zone, dict) else 0.0)),
            "level": float(sig["level"]),
            "idx": int(sig["idx"]),
            "entry": float(plan["entry"]),
            "stop": float(plan["stop"]),
            "tp1": float(plan["tp1"]),
            "atr": float(plan["atr"]),
            "score": float(score),
            "reasons": ",".join(reasons) if isinstance(reasons, (list, tuple)) else str(reasons),
            "is_backfill": bool(is_backfill),
            "dedupe_key": dedupe_key,
        }
        self._try(
            lambda: self.client.table(self.signals_table).insert(payload).execute(),
            tag=f"{self.signals_table}.insert",
        )
        print(f"[DB] signals insert OK ({self.signals_table}) {symbol} {timeframe} {sig['type']} {sig['direction']} score={score:.1f}")

    def update_signal(self, signal_id: int, fields: Dict[str, Any]) -> None:
        if not self.enabled or not self.client:
            return
        self._try(
            lambda: self.client.table(self.signals_table).update(fields).eq("id", signal_id).execute(),
            tag=f"{self.signals_table}.update",
        )

    # ----- outcome worker helper -----
    def fetch_recent_signals(self, since_iso: str, limit: int = 200) -> list[dict]:
        if not self.enabled or not self.client:
            return []
        res = self._try(
            lambda: self.client.table(self.signals_table)
            .select("*")
            .gte("created_at", since_iso)
            .order("created_at", desc=True)
            .limit(limit)
            .execute(),
            tag=f"{self.signals_table}.select",
        )
        return res.data or []
