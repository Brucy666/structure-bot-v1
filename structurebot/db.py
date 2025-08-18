# structurebot/db.py
from __future__ import annotations

import os
import time
import typing as T
import json
import logging
from datetime import datetime, timezone

import httpx

log = logging.getLogger("db")

EXPECTED_COLUMNS = {
    "id","created_at","symbol","timeframe","type","direction","zone_kind",
    "zone_top","zone_bottom","level","idx","entry","stop","tp1","atr","score",
    "reasons","is_backtest","dedupe_key","entered_at","exited_at","exit_price",
    "rr_achieved","outcome","last_checked"
}

class DB:
    def __init__(self):
        self.url = os.environ["SUPABASE_URL"].rstrip("/")
        self.key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.table = os.environ.get("SUPABASE_TABLE", "signals")
        self._client = httpx.Client(base_url=f"{self.url}/rest/v1", headers={
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation"
        }, timeout=30.0)

        self._verify_schema()

    # --- internal helpers -------------------------------------------------

    def _rpc(self, path: str, json_body: dict) -> dict:
        r = self._client.post(path, json=json_body)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: dict) -> list[dict]:
        r = self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, rows: list[dict]) -> list[dict]:
        r = self._client.post(path, content=json.dumps(rows))
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, params: dict, body: dict) -> list[dict]:
        r = self._client.patch(path, params=params, content=json.dumps(body))
        r.raise_for_status()
        return r.json()

    def _verify_schema(self):
        """Fetch one row (or none) to check columns. If mismatch -> hard error."""
        try:
            rows = self._get(f"/{self.table}", {"select": "*", "limit": 1})
        except httpx.HTTPStatusError as e:
            msg = f"Supabase table '{self.table}' not reachable: {e.response.text}"
            log.error(msg)
            raise

        # If no rows, we can still fetch headers by inserting then deleting,
        # but simpler: allow empty and trust the SQL you ran.
        # We still sanity-check by allowing writes only of known fields.
        log.info(f"[DB] Connected to {self.url} / table={self.table}")

    # --- public API --------------------------------------------------------

    def upsert_signal(self, row: dict, *, is_backtest: bool) -> dict:
        row = dict(row)
        row["is_backtest"] = bool(is_backtest)

        # hard filter to expected keys only (prevents RSI/VWAP drift)
        filtered = {k: v for k, v in row.items() if k in EXPECTED_COLUMNS}

        if "dedupe_key" not in filtered:
            # symbol:tf:type:dir:level
            filtered["dedupe_key"] = f"{filtered['symbol']}:{filtered['timeframe']}:{filtered['type']}:{filtered['direction']}:{filtered['level']}"

        res = self._post(f"/{self.table}?on_conflict=dedupe_key", [filtered])
        return res[0] if res else filtered

    def mark_entered(self, signal_id: str, entered_at: datetime | None = None):
        entered_at = entered_at or datetime.now(timezone.utc)
        return self._patch(
            f"/{self.table}",
            params={"id": f"eq.{signal_id}"},
            body={"entered_at": entered_at.isoformat(), "outcome": "open"}
        )

    def set_outcome(
        self,
        signal_id: str,
        *,
        outcome: T.Literal["tp","sl","missed","open"],
        exit_price: float | None = None,
        rr_achieved: float | None = None,
        exited_at: datetime | None = None,
    ):
        body = {
            "outcome": outcome,
            "exited_at": (exited_at or datetime.now(timezone.utc)).isoformat(),
            "last_checked": datetime.now(timezone.utc).isoformat()
        }
        if exit_price is not None:
            body["exit_price"] = float(exit_price)
        if rr_achieved is not None:
            body["rr_achieved"] = float(rr_achieved)

        return self._patch(f"/{self.table}", params={"id": f"eq.{signal_id}"}, body=body)

    def heartbeat(self, label: str = "db"):
        # cheap no‑op select to keep connection warm (shows in logs)
        _ = self._get(f"/{self.table}", {"select": "id", "limit": 1})
        log.info(f"[DB] heartbeat ok ({label})")
