# structurebot/db.py
from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("db")

# Only allow these fields to reach Supabase (prevents schema drift).
SIGNAL_COLUMNS = {
    "id", "created_at", "symbol", "timeframe",
    "type", "direction", "zone_kind",
    "zone_top", "zone_bottom", "level", "idx",
    "entry", "stop", "tp1", "atr", "score",
    "reasons", "is_backtest", "dedupe_key",
    "entered_at", "exited_at", "exit_price",
    "rr_achieved", "outcome", "last_checked",
    # add "minutes_in_trade" only if you've added the column in SQL
    # "minutes_in_trade",
}

ZONE_COLUMNS = {
    "id", "created_at", "symbol", "timeframe",
    "kind", "top", "bottom",
    "impulse_end_idx", "strength"
}


class DB:
    """
    Minimal Supabase REST client tailored for StructureBot.
    Uses Service Role key (server-side only).
    """

    def __init__(self) -> None:
        url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

        self.url = url
        self.key = key
        self.signals_table = os.environ.get("SUPABASE_TABLE", "signals")
        self.zones_table = os.environ.get("SUPABASE_ZONES_TABLE", "zones")

        self.client = httpx.Client(
            base_url=f"{self.url}/rest/v1",
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

        # Light connectivity check
        try:
            self._get(f"/{self.signals_table}", {"select": "id", "limit": 1})
            log.info(f"[DB] Connected to {self.url}  | signals={self.signals_table}, zones={self.zones_table}")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Supabase connection failed: {e}") from e

        self.enabled = True

    # ------------- low-level HTTP -----------------

    def _get(self, path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        r = self.client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _post_rows(self, path: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r = self.client.post(path, content=json.dumps(rows), headers={"Prefer": "return=representation"})
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return []

    def _patch(self, path: str, params: Dict[str, Any], body: Dict[str, Any]) -> List[Dict[str, Any]]:
        r = self.client.patch(path, params=params, content=json.dumps(body), headers={"Prefer": "return=representation"})
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return []

    # ------------- public helpers -----------------

    # Signals -----------------------------------------------------------------

    def log_signal(
        self,
        symbol: str,
        timeframe: str,
        sig: Dict[str, Any],
        zone: Any,
        plan: Dict[str, Any],
        score: float,
        reasons: List[str],
        is_backfill: bool,
        dedupe_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Insert (upsert-on-dedupe_key) one trading signal row.
        """
        if not self.enabled:
            return None

        # Accept zone as object or dict
        def zget(attr: str, default=None):
            if isinstance(zone, dict):
                return zone.get(attr, default)
            return getattr(zone, attr, default)

        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "type": sig.get("type"),
            "direction": sig.get("direction"),
            "zone_kind": zget("kind"),
            "zone_top": float(zget("top", 0.0)),
            "zone_bottom": float(zget("bottom", 0.0)),
            "level": float(sig.get("level", 0.0)),
            "idx": int(sig.get("idx", 0)),
            "entry": float(plan.get("entry", 0.0)),
            "stop": float(plan.get("stop", 0.0)),
            "tp1": float(plan.get("tp1", 0.0)),
            "atr": float(plan.get("atr", 0.0)),
            "score": float(score),
            "reasons": ", ".join(reasons) if isinstance(reasons, list) else str(reasons),
            "is_backtest": bool(is_backfill),
            "dedupe_key": dedupe_key,
            "outcome": None,
            "last_checked": None,
        }

        # hard filter to allowed keys (prevents stray cols like 'vwap', 'rsi')
        payload = {k: v for k, v in row.items() if k in SIGNAL_COLUMNS}

        try:
            out = self._post_rows(
                f"/{self.signals_table}?on_conflict=dedupe_key",
                [payload],
            )
            return out[0] if out else payload
        except httpx.HTTPError as e:
            log.error(f"[DB] log_signal error: {e} :: {getattr(e.response, 'text', '')}")
            return None

    def fetch_recent_signals(self, since_iso: str, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Pull recent signals (for outcome worker).
        """
        if not self.enabled:
            return []
        params = {
            "select": "*",
            "created_at": f"gte.{since_iso}",
            "order": "created_at.desc",
            "limit": max(1, min(limit, 1000)),
        }
        try:
            return self._get(f"/{self.signals_table}", params)
        except httpx.HTTPError as e:
            log.error(f"[DB] fetch_recent_signals error: {e}")
            return []

    def update_signal(self, signal_id: Any, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generic patch by id.
        """
        if not self.enabled:
            return None
        # filter allowed columns
        body = {k: v for k, v in fields.items() if k in SIGNAL_COLUMNS}
        try:
            out = self._patch(f"/{self.signals_table}", {"id": f"eq.{signal_id}"}, body)
            return out[0] if out else None
        except httpx.HTTPError as e:
            log.error(f"[DB] update_signal error: {e} :: {getattr(e.response, 'text', '')}")
            return None

    # Zones -------------------------------------------------------------------

    def upsert_zone(
        self,
        symbol: str,
        timeframe: str,
        kind: str,
        top: float,
        bottom: float,
        impulse_end_idx: int,
        strength: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Store latest detected zone snapshot.
        """
        if not self.enabled or not self.zones_table:
            return None

        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "kind": kind,
            "top": float(top),
            "bottom": float(bottom),
            "impulse_end_idx": int(impulse_end_idx),
            "strength": float(strength),
        }
        payload = {k: v for k, v in row.items() if k in ZONE_COLUMNS}

        try:
            out = self._post_rows(f"/{self.zones_table}", [payload])
            return out[0] if out else payload
        except httpx.HTTPError as e:
            # zones table is optional—log and continue
            log.warning(f"[DB] upsert_zone warn: {e} :: {getattr(e.response, 'text', '')}")
            return None

    # Convenience -------------------------------------------------------------

    def mark_entered(self, signal_id: Any, entered_at: Optional[datetime] = None):
        ts = (entered_at or datetime.now(timezone.utc)).isoformat()
        return self.update_signal(signal_id, {"entered_at": ts, "outcome": "open"})

    def set_outcome(
        self,
        signal_id: Any,
        *,
        outcome: str,  # "tp" | "sl" | "missed" | "open" | "timeout"
        exit_price: Optional[float] = None,
        rr_achieved: Optional[float] = None,
        exited_at: Optional[datetime] = None,
    ):
        body = {
            "outcome": outcome,
            "exited_at": (exited_at or datetime.now(timezone.utc)).isoformat(),
            "last_checked": datetime.now(timezone.utc).isoformat(),
        }
        if exit_price is not None:
            body["exit_price"] = float(exit_price)
        if rr_achieved is not None:
            body["rr_achieved"] = float(rr_achieved)
        return self.update_signal(signal_id, body)

    def heartbeat(self, label: str = "db") -> None:
        try:
            _ = self._get(f"/{self.signals_table}", {"select": "id", "limit": 1})
            log.info(f"[DB] heartbeat ok ({label})")
        except httpx.HTTPError as e:
            log.error(f"[DB] heartbeat failed: {e}")
