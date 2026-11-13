"""Data-driven period discovery on the training split (Algorithm 1 of the paper).

1. Detrend each channel with a centred moving average of width 2*floor(T/10)+1 and standardise.
2. a(l): autocorrelation averaged over channels (FFT).
3. Periodicity contrast c(l) = a(l) - a(floor(l/2)): high for a period (in phase at P,
   anti-phase at P/2), close to zero for a slow trend.
4. Candidates: local maxima of a with c >= theta and at least rho repetitions (l <= T/rho).
5. Fundamental P1: smallest candidate with c >= 0.8 * max c, refined by a comb score over harmonics.
6. Hierarchy: append the multiple m*P_k (m >= 2) with the highest a if a(m*P_k) >= a(P_k) + gamma.

Returns [1] (no periodicity) when no lag qualifies. The nested prefixes of the returned
list are the candidates among which the validation loss chooses.
"""

import numpy as np


def _detrend(x, width):
    if width is None or width <= 1 or width >= len(x):
        return x
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), 0)
    h = width // 2
    idx = np.arange(len(x))
    lo, hi = np.clip(idx - h, 0, len(x)), np.clip(idx + h + 1, 0, len(x))
    return x - (c[hi] - c[lo]) / (hi - lo)[:, None]


def acf_mean(x, max_lag, detrend_width=None):
    """Channel-averaged autocorrelation a(0..max_lag+1) of a (T, C) array."""
    x = _detrend(np.asarray(x, dtype=np.float64), detrend_width)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    T = len(x)
    n = 1 << int(np.ceil(np.log2(2 * T)))
    f = np.fft.rfft(x, n=n, axis=0)
    ac = np.fft.irfft(f * np.conj(f), n=n, axis=0)[: max_lag + 2] / T
    return ac.mean(1)


def discover_periods(x, theta=0.05, rho=10, gamma=0.0, k_max=3, min_lag=4, comb=True):
    """Propose a nested period hierarchy from a (T, C) training array.

    Returns (periods, a) where periods is e.g. [24, 168] or [1], and a is the ACF used.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    T = len(x)
    max_lag = max(min_lag + 2, T // rho)
    a = acf_mean(x, max_lag, detrend_width=2 * (T // 10) + 1)

    def contrast(l):
        return a[l] - a[l // 2]

    peaks = [
        l
        for l in range(min_lag, max_lag)
        if a[l] >= a[l - 1] and a[l] >= a[l + 1] and contrast(l) >= theta
    ]
    if not peaks:
        return [1], a
    best = max(contrast(l) for l in peaks)
    p1 = min(l for l in peaks if contrast(l) >= 0.8 * best)
    if comb:
        window = range(
            max(min_lag, int(p1 * 0.95)), min(max_lag, int(np.ceil(p1 * 1.05))) + 1
        )

        def comb_score(l):
            n_harm = max(2, min(6, max_lag // l))
            return np.mean([a[m * l] for m in range(1, n_harm + 1) if m * l <= max_lag])

        p1 = max(window, key=comb_score)
    periods = [int(p1)]
    while len(periods) < k_max:
        p = periods[-1]
        multiples = [l for l in range(2 * p, max_lag, p) if a[l] >= a[p] + gamma]
        if not multiples:
            break
        periods.append(int(max(multiples, key=lambda l: a[l])))
    return periods, a
