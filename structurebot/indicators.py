import numpy as np

def true_range(o, h, l, c):
    prev_close = np.r_[np.nan, c[:-1]]
    v1 = h - l
    v2 = np.abs(h - prev_close)
    v3 = np.abs(l - prev_close)
    tr = np.nanmax(np.vstack([v1, v2, v3]), axis=0)
    return tr

def atr(o, h, l, c, length=14):
    tr = true_range(o, h, l, c)
    alpha = 2/(length+1)
    atr_vals = np.full_like(tr, np.nan, dtype=float)
    for i, x in enumerate(tr):
        if np.isnan(x):
            continue
        if i == 0 or np.isnan(atr_vals[i-1]):
            atr_vals[i] = x
        else:
            atr_vals[i] = alpha * x + (1 - alpha) * atr_vals[i-1]
    return atr_vals
