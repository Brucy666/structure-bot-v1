# structurebot/db.py
from __future__ import annotations

import os
import json
import time
import typing as T
from datetime import datetime, timezone, timedelta
from collections import OrderedDict

import httpx


# Only fields we allow into Supabase (prevents stray columns like vwap/rsi, etc.)
_ALLOWED_FIELDS = {
    "id","created_at","symbol","timeframe","type","direction","zone_kind",
    "zone_top","zone_bottom","level","idx","entry","stop","tp1","atr","score",
    "reasons","is_backtest","dedupe_key","entered_at","exited_at","exit_price",
    "rr_achieved","outcome","minutes_in_trade","last_checked"
}


class _TTLCache:
    """Tiny in‑memory TTL cache for dedupe_keys to reduce duplicate writes."""
    def __init__(self, maxlen: int = 5000, ttl_seconds: int = 3600):
        self._store: "OrderedDict[str, float]" = OrderedDict()
        self.maxlen = maxlen
        self.ttl = float(ttl_seconds)

    def add(self, key: str) -> None:
        now = time.time()
        self._store[key] = now
        self._store.move_to_end(key)
        self._prune(now)

    def seen(self, key: str) -> bool:
        ts = self._store.get(key)
        if ts is None:
            return False
        if (time.time() - ts) > self.ttl:
            # expired
            self._store.pop(key, None)
            return False
        return True

    def _prune(self, now: float) -> None:
        # size bound
        while len(self._store) > self.maxlen:
            self._store.popitem(last=False)
        # TTL sweep (cheap)
        drop = [k for k, ts in self._store.items() if (now - ts) > self.ttl]
        for k in drop:
            self._store.pop(k, None)


class DB:
    """Very small Supabase REST client with safe upserts + local dedupe."""
    def __init__(self) -> None:
        url = os.getenv("SUPABASE_URL", "").rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        table = os.getenv("SUPABASE_TABLE", "signals")
        if not url or not key:
            # allow the app to run without DB (e.g., local testing)
            self.enabled = False
            self.url = ""
            self.key = ""
            self.table = table
            self._client = None
            self._cache = _TTLCache()
            return

        self.enabled = True
        self.url = url
        self.key = key
        self.table = table

        self._client = httpx.Client(
            base_url=f"{self.url}/rest/v1",
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # This is the key line for merging duplicates server‑side
                "Prefer": "resolution=merge-duplicates,return=representation",
            },
            timeout=30.0,
        )
        self._cache = _TTLCache()

    # ----------------- low-level -----------------

    def _get(self, path: str, params: dict | None = None) -> list[dict]:
        if not self.enabled:
            return []
        r = self._client.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, rows: list[dict]) -> list[dict]:
        if not self.enabled:
            return []
        # Ensure query params include on_conflict + return=representation
        if "?" in path:
            path = f"{path}&return=representation"
        else:
            path = f"{path}?return=representation"
        r = self._client.post(path, content=json.dumps(rows))
        # If we still ever hit 409 due to race, treat as non‑fatal
        if r.status_code == 409:
            return []
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, params: dict, body: dict) -> list[dict]:
        if not self.enabled:
            return []
        r = self._client.patch(path, params=params, content=json.dumps(body))
        r.raise_for_status()
        return r.json()

    # ----------------- public API -----------------

    def log_signal(
        self,
        symbol: str,
        timeframe: str,
        sig: dict,
        zone: T.Any,
        plan: dict,
        score: float,
        reasons: list[str] | None,
        is_backtest: bool,
        dedupe_key: str,
    ) -> None:
        """Insert/merge a signal row (de‑duped locally + server side)."""
        if not self.enabled:
            print("[DB] disabled — skipping insert")
            return

        # local dedupe (cheap)
        if self._cache.seen(dedupe_key):
            return
        self._cache.add(dedupe_key)

        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),

            "symbol": symbol,
            "timeframe": timeframe,

            "type": sig.get("type"),
            "direction": sig.get("direction"),

            "zone_kind": "bullish" if sig.get("direction") == "bullish" else "bearish",
            "zone_top": float(getattr(zone, "top", sig.get("zone_top", 0.0))),
            "zone_bottom": float(getattr(zone, "bottom", sig.get("zone_bottom", 0.0))),

            "level": float(sig.get("level", 0.0)),
            "idx": int(sig.get("idx", 0)),

            "entry": float(plan.get("entry", 0.0)),
            "stop": float(plan.get("stop", 0.0)),
            "tp1": float(plan.get("tp1", 0.0)),
            "atr": float(plan.get("atr", 0.0)),
            "score": float(score),

            "reasons": ", ".join(reasons or []),

            "is_backtest": bool(is_backtest),
            "dedupe_key": dedupe_key,
        }

        # strict allow‑list
        row = {k: v for k, v in row.items() if k in _ALLOWED_FIELDS}

        # server‑side merge on dedupe_key
        self._post(f"/{self.table}?on_conflict=dedupe_key", [row])

    def upsert_signal(self, row: dict, *, is_backtest: bool) -> None:
        """Generic upsert if a caller already built the row."""
        if not self.enabled:
            return
        row = dict(row)
        row["is_backtest"] = bool(is_backtest)
        if "dedupe_key" not in row:
            # fallback: symbol:tf:type:dir:level (rounded)
            row["dedupe_key"] = f"{row['symbol']}:{row['timeframe']}:{row['type']}:{row['direction']}:{round(float(row['level']),6)}"
        key = row["dedupe_key"]
        if self._cache.seen(key):
            return
        self._cache.add(key)

        row = {k: v for k, v in row.items() if k in _ALLOWED_FIELDS}
        self._post(f"/{self.table}?on_conflict=dedupe_key", [row])

    def update_signal(self, signal_id: str, body: dict) -> None:
        if not self.enabled:
            return
        self._patch(f"/{self.table}", params={"id": f"eq.{signal_id}"}, body=body)

    def fetch_recent_signals(self, since_iso: str, limit: int = 200) -> list[dict]:
        if not self.enabled:
            return []
        return self._get(
            f"/{self.table}",
            params={
                "select": "*",
                "created_at": f"gte.{since_iso}",
                "order": "created_at.desc",
                "limit": str(int(limit)),
            },
        )

    # Optional helpers the scanners/worker use

    def upsert_zone(self, symbol: str, timeframe: str, kind: str,
                    top: float, bottom: float, impulse_end_idx: int, strength: float) -> None:
        """If you have a zones table, write it here (safe no‑op otherwise)."""
        if not self.enabled:
            return
        try:
            payload = [{
                "symbol": symbol,
                "timeframe": timeframe,
                "kind": kind,
                "top": float(top),
                "bottom": float(bottom),
                "impulse_end_idx": int(impulse_end_idx),
                "strength": float(strength),
                "dedupe_key": f"{symbol}:{timeframe}:{kind}:{round(top,6)}:{round(bottom,6)}:{impulse_end_idx}"
            }]
            self._post("/zones?on_conflict=dedupe_key", payload)
        except Exception:
            # zones table is optional; swallow errors to avoid breaking the bot
            pass
