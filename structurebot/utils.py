from typing import List
from .structure_engine import Candle

def to_candles(ohlcv: list) -> List[Candle]:
    # ohlcv: [ [ts, open, high, low, close, volume], ... ]
    return [
        Candle(ts=int(x[0]), o=float(x[1]), h=float(x[2]), l=float(x[3]), c=float(x[4]), v=float(x[5]))
        for x in ohlcv
    ]
