import ccxt

class DataFeed:
    def __init__(self, exchange_name: str):
        ex = getattr(ccxt, exchange_name)()
        ex.enableRateLimit = True
        self.exchange = ex

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        # returns list of [ts, open, high, low, close, volume]
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
