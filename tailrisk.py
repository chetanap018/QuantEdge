"""
tailrisk.py
-----------
Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall) reporting.

Computed on per-trade net-PnL series (preferred) or on equity-curve
returns -- both are one-liners once the engine hands over the series.
Sign convention: losses are NEGATIVE numbers; VaR is reported as a
positive loss magnitude plus the signed quantile for precision.
"""

from typing import Dict, Sequence

import numpy as np


def _as_array(pnls: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(pnls), dtype=float)
    return arr[~np.isnan(arr)]


def var_historical(pnls: Sequence[float], confidence: float = 0.95) -> float:
    """
    Historical VaR: the loss that is NOT exceeded (1-confidence) of the time.

    Returns a POSITIVE loss magnitude (0.0 when no losses observed).
    """
    arr = _as_array(pnls)
    if arr.size == 0:
        return 0.0
    q = float(np.quantile(arr, 1.0 - confidence))
    return round(abs(q) if q < 0 else 0.0, 2)


def cvar_historical(pnls: Sequence[float], confidence: float = 0.95) -> float:
    """
    Historical Conditional VaR (Expected Shortfall): mean loss GIVEN that
    the loss exceeds VaR.  Positive loss magnitude, >= VaR by construction.
    """
    arr = _as_array(pnls)
    if arr.size == 0:
        return 0.0
    cutoff = float(np.quantile(arr, 1.0 - confidence))
    tail = arr[arr <= cutoff]
    if tail.size == 0:
        return round(abs(cutoff) if cutoff < 0 else 0.0, 2)
    return round(abs(float(tail.mean())) if float(tail.mean()) < 0 else 0.0, 2)


def var_parametric(pnls: Sequence[float], confidence: float = 0.95) -> float:
    """Gaussian VaR = -(mu - z * sigma); positive loss magnitude."""
    from math import erf

    arr = _as_array(pnls)
    if arr.size < 2:
        return 0.0
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    # inverse normal CDF via rational approximation (Acklam)
    p = 1.0 - confidence
    z = _ndtri(p)
    var_signed = mu + z * sigma
    return round(abs(var_signed) if var_signed < 0 else 0.0, 2)


def _ndtri(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = (-2 * _log(p)) ** 0.5
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = (-2 * _log(1 - p)) ** 0.5
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _log(x: float) -> float:
    import math
    return math.log(x)


def tail_report(pnls: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """One-call summary: {'var_95': ..., 'cvar_95': ..., 'var_param_95': ...}."""
    tag = str(int(round(confidence * 100)))
    return {
        f"var_{tag}": var_historical(pnls, confidence),
        f"cvar_{tag}": cvar_historical(pnls, confidence),
        f"var_param_{tag}": var_parametric(pnls, confidence),
    }



def results_with_tail(results: dict, net_pnls) -> dict:
    """Return `results` with VaR_95 / CVaR_95 keys merged in."""
    out = dict(results)
    try:
        out.update(tail_report(list(net_pnls or []), 0.95))
    except Exception:
        pass
    return out

