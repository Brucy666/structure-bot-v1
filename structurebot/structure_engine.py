from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from .indicators import atr

@dataclass
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float
    v: float

@dataclass
class Impulse:
    direction: str   # 'up' or 'down'
    start_idx: int
    end_idx: int
    body_fraction: float
    range_points: float

@dataclass
class Zone:
    kind: str        # 'bullish' (from up impulse) or 'bearish' (from down impulse)
    bottom: float
    top: float
    impulse_end_idx: int

class StructureEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def _to_arrays(self, candles: List[Candle]):
        o = np.array([x.o for x in candles], float)
        h = np.array([x.h for x in candles], float)
        l = np.array([x.l for x in candles], float)
        c = np.array([x.c for x in candles], float)
        t = np.array([x.ts for x in candles], np.int64)
        return t, o, h, l, c

    def detect_last_impulse(self, candles: List[Candle]) -> Optional[Impulse]:
        if len(candles) < 5:
            return None
        t, o, h, l, c = self._to_arrays(candles)
        length = self.cfg['impulse']['atr_len']
        body_min = self.cfg['impulse']['body_min']
        atr_mult = self.cfg['impulse']['atr_mult']
        min_consec = self.cfg['impulse']['min_consecutive']

        a = atr(o, h, l, c, length)
        bodies = np.abs(c - o)
        trange = h - l
        trange_safe = np.where(trange == 0, 1e-9, trange)
        body_frac = bodies / trange_safe
        direction = np.where(c >= o, 1, -1)

        # scan backward for most recent run of impulsive candles
        i = len(c) - 1
        end = i
        consec = 1

        def candle_ok(idx):
            if idx < 0 or idx >= len(c):
                return False
            bf = body_frac[idx]
            tr_ok = (h[idx] - l[idx]) >= atr_mult * (a[idx] if not np.isnan(a[idx]) else 0)
            return (bf >= body_min) and tr_ok

        # ensure last candle meets criteria
        if not candle_ok(end):
            # try one step back (sometimes the last bar is small)
            end -= 1
            if end < 0 or not candle_ok(end):
                return None

        i = end
        while i > 0 and direction[i] == direction[i-1] and candle_ok(i-1):
            consec += 1
            i -= 1

        if consec < min_consec:
            return None

        start = i
        dir_str = 'up' if direction[end] == 1 else 'down'
        rng = (h[end] - l[end])
        return Impulse(
            direction=dir_str,
            start_idx=start,
            end_idx=end,
            body_fraction=float(np.nanmean(body_frac[start:end+1])),
            range_points=float(rng)
        )

    def make_zone_from_impulse(self, candles: List[Candle], impulse: Impulse) -> Optional[Zone]:
        if impulse is None:
            return None
        last = candles[impulse.end_idx]
        if impulse.direction == 'up':
            wick_low = last.l
            body_low = min(last.o, last.c)
            bottom, top = wick_low, body_low
            kind = 'bullish'
        else:
            wick_high = last.h
            body_high = max(last.o, last.c)
            bottom, top = body_high, wick_high
            kind = 'bearish'

        max_pct = self.cfg['zones']['max_zone_pct']
        imp_range = max(impulse.range_points, 1e-9)
        if (top - bottom) / imp_range > max_pct:
            mid = (top + bottom) / 2
            half = imp_range * max_pct / 2
            bottom, top = mid - half, mid + half

        return Zone(kind=kind, bottom=float(bottom), top=float(top), impulse_end_idx=impulse.end_idx)

    def bos_signal(self, candles: List[Candle], zone: Zone):
        if zone is None:
            return None
        confirm = self.cfg['signals']['confirm_closes']
        closes = [x.c for x in candles]

        if zone.kind == 'bearish':
            idxs = [i for i in range(zone.impulse_end_idx + 1, len(candles)) if closes[i] > zone.top]
            if len(idxs) >= confirm:
                return {"type": "BOS", "direction": "bullish", "level": zone.top, "idx": idxs[-1]}
        else:
            idxs = [i for i in range(zone.impulse_end_idx + 1, len(candles)) if closes[i] < zone.bottom]
            if len(idxs) >= confirm:
                return {"type": "BOS", "direction": "bearish", "level": zone.bottom, "idx": idxs[-1]}
        return None

    def sfp_signal(self, candles: List[Candle], zone: Zone):
        if zone is None:
            return None
        w = self.cfg['signals']['sfp_window']
        if w <= 0:
            return None

        closes = [x.c for x in candles]
        highs  = [x.h for x in candles]
        lows   = [x.l for x in candles]
        start = zone.impulse_end_idx + 1

        if zone.kind == 'bearish':
            for i in range(start, len(candles)):
                if highs[i] > zone.top and closes[i] <= zone.top:
                    return {"type": "SFP", "direction": "bearish-failed-break", "level": zone.top, "idx": i}
                for j in range(i+1, min(i+1+w, len(candles))):
                    if highs[i] > zone.top and closes[j] <= zone.top:
                        return {"type": "SFP", "direction": "bearish-failed-break", "level": zone.top, "idx": j}
        else:
            for i in range(start, len(candles)):
                if lows[i] < zone.bottom and closes[i] >= zone.bottom:
                    return {"type": "SFP", "direction": "bullish-failed-break", "level": zone.bottom, "idx": i}
                for j in range(i+1, min(i+1+w, len(candles))):
                    if lows[i] < zone.bottom and closes[j] >= zone.bottom:
                        return {"type": "SFP", "direction": "bullish-failed-break", "level": zone.bottom, "idx": j}
        return None
