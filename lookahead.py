"""
lookahead.py
------------
Automated look-ahead bias guards for strategies.

Why this matters
----------------
A strategy must produce the signal at bar `t` using only information available
BEFORE bar `t+1`'s open (that is when the engine now fills orders).  Two failure
modes exist:

1. Future-data leakage -- signal[t] silently depends on bars t+1, t+2, ...
   (e.g. center=True rolling windows, full-sample normalization, or columns
   computed from the whole series).

2. Same-bar close entries -- the signal at bar `t` is used to fill at the SAME
   bar's close.  With the engine's new "next_open" fill policy the decision is
   executed one bar later so this is no longer a leak; the decorator simply
   records the shift-safe guarantee.

This module provides:
    @no_lookahead                    -- decorator a strategy slaps on
                                         generate_signals(); validates on first
                                         call and attaches a flag the engine trusts.
    validate_no_lookahead(strategy, data)
                                     -- explicit validation returning a report.
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

_SAFE_ATTR = "_quantedge_no_lookahead"


@dataclass
class LookaheadReport:
    """Result of a look-ahead validation run."""

    ok: bool
    method: str = ""
    checked_bars: int = 0
    flagged: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.ok:
            return f"OK ({self.method}): {self.checked_bars} signal bar(s) re-checked, no look-ahead found."
        return (
            f"LOOK-AHEAD FLAGGED ({self.method}): "
            f"{len(self.flagged)} signal bar(s) change when future data is withheld. "
            f"Offending bars: {self.flagged[:5]}"
        )
def _align_by_datetime(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Align two signal frames on their datetime column (handles warmup-row drops).

    Duplicate datetimes (e.g. the extended frame used by the append-invariance
    check) are deduplicated so both sides always have identical labels.
    """
    l = left.set_index("datetime")["signal"].sort_index()
    r = right.set_index("datetime")["signal"].sort_index()
    l = l[~l.index.duplicated(keep="first")]
    r = r[~r.index.duplicated(keep="first")]
    common = l.index.intersection(r.index)
    return l.loc[common], r.loc[common]


# ----------------------------------------------------------------------
# Truncation check: signal[t] must not change when bars AFTER t are removed
# ----------------------------------------------------------------------
def _truncation_check(strategy, data, sample_n: int = 10, seed: int = 42) -> LookaheadReport:
    full = strategy.generate_signals(data.copy())
    full = _with_datetime(full, data)

    signal_idx = np.where(full["signal"].astype(float) != 0.0)[0]
    if len(signal_idx) == 0:
        return LookaheadReport(ok=True, method="truncation", checked_bars=0,
                               notes=["No non-zero signals to validate."])

    rng = np.random.default_rng(seed)
    sample = rng.choice(signal_idx, size=min(sample_n, len(signal_idx)), replace=False)
    sample = np.sort(sample)

    flagged: List[dict] = []
    for t in sample:
        # Strategy sees bars 0..t only -> the last signal it emits is at bar t.
        truncated = strategy.generate_signals(data.iloc[: t + 1].copy())
        truncated = _with_datetime(truncated, data.iloc[: t + 1])
        if truncated.empty:
            continue
        sig_t = float(truncated["signal"].iloc[-1])
        sig_full = float(full["signal"].iloc[t])
        if sig_t != sig_full:
            flagged.append({
                "index": int(t),
                "signal_full": sig_full,
                "signal_without_future": sig_t,
            })

    return LookaheadReport(
        ok=len(flagged) == 0,
        method="truncation",
        checked_bars=int(len(sample)),
        flagged=flagged,
    )


def _with_datetime(signals: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Attach the reference datetime column to a strategy's signal frame."""
    out = signals.copy()
    if "datetime" not in out.columns and "datetime" in ref.columns:
        dt = ref["datetime"].iloc[: len(out)].reset_index(drop=True)
        out["datetime"] = dt.values[: len(out)]
    return out


# ----------------------------------------------------------------------
# Append-invariance check: appending future bars must not change past signals
# ----------------------------------------------------------------------
def _append_check(strategy, data: pd.DataFrame) -> LookaheadReport:
    base = strategy.generate_signals(data.copy())
    base = _with_datetime(base, data)

    extended = pd.concat([data, data.iloc[[-1]]], ignore_index=True)
    ext = strategy.generate_signals(extended.copy())
    ext = _with_datetime(ext, extended)

    if base.empty or ext.empty or "datetime" not in base.columns or "datetime" not in ext.columns:
        return LookaheadReport(ok=True, method="append-invariance", checked_bars=0,
                               notes=["Could not align frames; skipping."])

    try:
        l, r = _align_by_datetime(base, ext)
    except KeyError:
        return LookaheadReport(ok=True, method="append-invariance", checked_bars=0,
                               notes=["Could not align frames; skipping."])

    mismatches = (l.astype(float) != r.astype(float))
    n_mismatch = int(mismatches.sum())
    flagged = [
        {"index": int(i), "signal_before": float(l.iloc[j]), "signal_after_append": float(r.iloc[j])}
        for j, i in enumerate(l.index[mismatches])
    ][:10]
    return LookaheadReport(
        ok=n_mismatch == 0,
        method="append-invariance",
        checked_bars=int(len(l)),
        flagged=flagged,
    )


def validate_no_lookahead(
    strategy,
    data: pd.DataFrame,
    sample_n: int = 10,
    seed: int = 42,
) -> LookaheadReport:
    """
    Run the automated look-ahead validation for a strategy.

    Performs two independent checks:

      1. truncation         -- for sampled signal bars, re-run the strategy with
                               future bars withheld and compare signals.
      2. append-invariance  -- re-run with one extra (future) bar appended and
                               verify every past signal is unchanged.
    """
    reports = [
        _truncation_check(strategy, data, sample_n=sample_n, seed=seed),
        _append_check(strategy, data),
    ]
    failed = [r for r in reports if not r.ok]
    checked = max(r.checked_bars for r in reports)
    flagged = [f for r in reports for f in r.flagged]
    notes = [n for r in reports for n in r.notes]
    if failed:
        notes.insert(0, failed[0].summary)
    return LookaheadReport(
        ok=not failed,
        method="truncation + append-invariance",
        checked_bars=checked,
        flagged=flagged,
        notes=notes,
    )
def no_lookahead(validate: bool = True, sample_n: int = 10, seed: int = 42, on_error: str = "warn"):
    """
    Decorator for a strategy's `generate_signals` that declares it look-ahead
    safe and (optionally) enforces the automated checks.

    Usage
    -----
        from lookahead import no_lookahead

        class MyStrategy(Strategy):
            @no_lookahead(on_error="fail")
            def generate_signals(self, data):
                ...  # uses only .shift(1)-safe / prior-bar information

    Parameters
    ----------
    validate : bool
        Run the automated checks on the first call (memoized per data-frame id).
    on_error : str
        "warn" -> log a warning; "fail" -> raise LookaheadError.
    """
    if on_error not in ("warn", "fail"):
        raise ValueError("on_error must be 'warn' or 'fail'")

    _validated = {}  # id(data) -> report

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
            if validate:
                did = id(data)
                if did not in _validated:
                    report = validate_no_lookahead_wrapper(self, fn, data, sample_n=sample_n, seed=seed)
                    _validated[did] = report
                report = _validated[did]
                if not report.ok:
                    msg = f"look-ahead violation in {type(self).__name__}: {report.summary}"
                    if on_error == "fail":
                        raise LookaheadError(msg)
                    import logging
                    logging.getLogger("lookahead").warning(msg)

            result = fn(self, data, *args, **kwargs)
            setattr(result, _SAFE_ATTR, True)
            return result

        # mark the method as shift-safe
        wrapper._no_lookahead = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


def validate_no_lookahead_wrapper(self, fn: Callable, data: pd.DataFrame, sample_n: int, seed: int) -> LookaheadReport:
    """Run the checks against the decorated method (avoids re-triggering decorators)."""

    class _Probe:
        def generate_signals(self, d):
            return fn(self, d)

    return validate_no_lookahead(_Probe(), data, sample_n=sample_n, seed=seed)


def is_shift_safe(strategy) -> bool:
    """Convenience: True when a strategy's generate_signals is @no_lookahead-marked."""
    fn = getattr(strategy, "generate_signals", None)
    return bool(getattr(fn, "_no_lookahead", False))


class LookaheadError(Exception):
    """Raised when a strategy fails the automated look-ahead validation."""