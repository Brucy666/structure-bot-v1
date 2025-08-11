import json
import time
from pathlib import Path

class State:
    def __init__(self, path: str = "structure_state.json"):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text(json.dumps({}))
        self._load()

    def _load(self):
        self.data = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def can_alert(self, key: str, dedupe_minutes: int) -> bool:
        now = int(time.time())
        last = self.data.get(key, 0)
        if now - last >= dedupe_minutes * 60:
            self.data[key] = now
            self.save()
            return True
        return False
