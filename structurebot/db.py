# structurebot/db.py
import os
import time
from typing import Any, Callable, Dict, Optional, List

from supabase import create_client, Client


class DB:
    """
    Thin Supabase helper for StructureBot.

    - Env-driven table name for signals: SB_SIGNALS_TABLE (default: 'signals')
    - Safe retries around calls
    - Backward-compatible upsert_zone()  -> supports both old (3-arg) and new (7-arg) usage
    """

    def __init__(self) -> None:
        self.url: str = os.getenv("SUPABASE_URL", "")
        self.key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.signals_table: str = os.getenv("SB_SIGNALS_TABLE", "signals")

        self.enabled: bool = bool(self.url and self.key)
        self.client: Optional[Client] = None

        if self.enabled:
            self.client = create_client(self.url, self.key)
            print(f"[DB] Connected to {self.url} | signals_table={self.signals_table}")
        else:
            print("[DB] Supabase disabled (missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY)")

    # ---------- internals ----------
    def _try(self, fn: Callable[[], Any], tag: str, tries: int = 3, delay: float = 0.6) -> Any:
        last_exc = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last_exc = e
                print(f"[DB] {tag} error ({i+1}/{tries}): {e}")
                time.sleep(delay)
        raise last_exc

    # ---------- zones ----------
    def upsert_zone(
        self,
        symbol: str,
        timeframe: str,
        kind: str,
        top: float = None,
        bottom: float = None,
        impulse_end_idx: int = None,
        strength: float = None,
    ) -> None:
        """
        Backward-compatible upsert.
        - If only (symbol, timeframe, kind) are provided, we warn and skip write (no crash).
        - If full args provided, we upsert unique (symbol,timeframe,kind,top,bottom).
        """
        if not self.enabled or not self.client:
            return

        if top is None or bottom is None:
            # Old callsite signature; don't blow up the worker.
            print(f"[DB] WARN upsert_zone called without full args: {symbol} {timeframe} {kind} -> skipped")
            return

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "kind": kind,
            "top": float(top),
            "bottom": float(bottom),
            "impulse_end_idx": int(impulse_end_idx) if impulse_end_idx is not None else 0,
            "strength": float(strength) if strength is not None else 0.0,
        }
        self._try(
            lambda: self.client.table("zones")
            .upsert(payload, on_conflict="symbol,timeframe,kind,top,bottom")
            .execute(),
            tag="zones.upsert",
        )

    # ---------- signals ----------
    def log_signal(
        self,
        symbol: str,
        timeframe: str,
        sig: Dict[str, Any],      # expects: type, direction, level, idx
        zone: Any,                # object with .kind .top .bottom (or dict)
        plan: Dict[str, float],   # expects: entry, stop, tp1, atr
        score: float,
        reasons: Any,
        is_backfill: bool,
        dedupe_key: str,
    ) -> None:
        if not self.enabled or not self.client:
            return

        z_kind = getattr(zone, "kind", zone.get("kind") if isinstance(zone, dict) else None)
        z_top = getattr(zone, "top", zone.get("top") if isinstance(zone, dict) else 0.0)
        z_bot = getattr(zone, "bottom", zone.get("bottom") if isinstance(zone, dict) else 0.0)

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "type": sig["type"],
            "direction": sig["direction"],
            "zone_kind": z_kind,
            "zone_top": float(z_top),
            "zone_bottom": float(z_bot),
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
        print(
            f"[DB] signals insert OK ({self.signals_table}) "
            f"{symbol} {timeframe} {sig['type']} {sig['direction']} score={score:.1f}"
        )

    def update_signal(self, signal_id: int, fields: Dict[str, Any]) -> None:
        if not self.enabled or not self.client:
            return
        self._try(
            lambda: self.client.table(self.signals_table).update(fields).eq("id", signal_id).execute(),
            tag=f"{self.signals_table}.update",
        )

    # ---------- outcome worker helper ----------
    def fetch_recent_signals(self, since_iso: str, limit: int = 200) -> List[Dict[str, Any]]:
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
