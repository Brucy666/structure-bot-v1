import requests

class Notifier:
    def __init__(self, webhook_url: str):
        self.url = webhook_url

    def post(self, payload):
        if not self.url:
            print("[Notifier] No webhook configured. Payload:", payload)
            return
        try:
            requests.post(self.url, json=payload, timeout=10)
        except Exception as e:
            print("[Notifier] Post error:", e)
