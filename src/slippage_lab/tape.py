"""A synthetic intraday tape: a price path and a volume profile, binned.

Nothing here is market data. The tape exists so that the rest of the library
has something honest to measure against -- a price path whose drift we chose,
and a volume profile with roughly the shape intraday volume actually has.

Two prices are kept per bin, and the distinction matters downstream:

* ``bin_open_mid`` is the mid at the *boundary* of each bin. Arrival price and
  delay cost are measured off boundary mids, because those are instants, and a
  decision happens at an instant.
* ``bin_vwap`` is the volume-weighted price of market trades *within* the bin.
  Fills and the interval VWAP benchmark are measured off these, because those
  are averages over an interval.

Mixing the two up is a common and quiet source of error in cost analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Expected total drift over the session, in basis points, per named regime.
#: "trending" is deliberately strong: an adverse trend is the condition under
#: which the two benchmarks have anything to disagree about.
REGIMES: dict[str, float] = {"flat": 0.0, "trending": 250.0}


@dataclass(frozen=True)
class Tape:
    """One session of synthetic market activity, binned into equal intervals.

    Volumes are *other participants'* volume. Our own fills are never added to
    the tape, which is what lets ``benchmarks.interval_vwap`` be correct by
    construction rather than by remembering to subtract something.
    """

    bin_vwap: FloatArray
    """Volume-weighted price of market trades inside each bin."""

    bin_volume: FloatArray
    """Market volume traded inside each bin, excluding our own order."""

    bin_open_mid: FloatArray
    """Mid price at the instant each bin opens."""

    close_mid: float
    """Mid price at the instant the session closes."""

    bin_minutes: float
    """Wall-clock length of one bin, for reporting only."""

    def __post_init__(self) -> None:
        n = self.bin_vwap.size
        if n == 0:
            raise ValueError("a tape needs at least one bin")
        if self.bin_volume.size != n or self.bin_open_mid.size != n:
            raise ValueError("bin_vwap, bin_volume and bin_open_mid must be the same length")
        if not np.all(self.bin_volume > 0.0):
            raise ValueError("every bin must have positive market volume")
        if not np.all(self.bin_vwap > 0.0) or not np.all(self.bin_open_mid > 0.0):
            raise ValueError("prices must be positive")

    @property
    def n_bins(self) -> int:
        return int(self.bin_vwap.size)

    @property
    def decision_mid(self) -> float:
        """Mid at the decision instant, i.e. the open of the first bin."""
        return float(self.bin_open_mid[0])

    @property
    def total_volume(self) -> float:
        return float(self.bin_volume.sum())


def _volume_profile(n_bins: int, *, midday_floor: float) -> FloatArray:
    """U-shaped intraday volume weights: heavy at the open and close, light midday.

    A quadratic in time-of-day is enough. The real curve is not a parabola, but
    it is U-shaped, and the U is the only feature any of the downstream maths
    depends on.
    """
    u = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    weights = midday_floor + (1.0 - midday_floor) * (2.0 * u - 1.0) ** 2
    return weights / weights.sum()


def build_tape(
    *,
    n_bins: int = 26,
    session_minutes: float = 390.0,
    start_price: float = 100.0,
    drift_bps: float = 0.0,
    vol_bps: float = 60.0,
    session_volume: float = 8_000_000.0,
    seed: int = 0,
    steps_per_bin: int = 8,
    volume_noise: float = 0.15,
    midday_floor: float = 0.25,
) -> Tape:
    """Build a deterministic synthetic tape.

    The price path is geometric Brownian motion. ``drift_bps`` is the *expected*
    total log drift over the whole session and ``vol_bps`` the standard
    deviation of the total log return, both quoted in basis points, because
    those are the units a trader thinks in. Realised drift differs from
    expected drift by the noise, as it should.

    The path is simulated on a finer grid than the bins (``steps_per_bin``) so
    that ``bin_vwap`` is a genuine within-bin average rather than a point
    sample. Sub-bin volume is taken as uniform, so the within-bin VWAP is the
    arithmetic mean of the sub-step prices.

    Same ``seed`` gives byte-identical output, always.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if steps_per_bin < 1:
        raise ValueError("steps_per_bin must be at least 1")
    if session_volume <= 0.0:
        raise ValueError("session_volume must be positive")

    rng = np.random.default_rng(seed)
    n_steps = n_bins * steps_per_bin

    total_drift_log = np.log1p(drift_bps / 10_000.0)
    total_vol_log = vol_bps / 10_000.0
    step_drift = total_drift_log / n_steps
    step_vol = total_vol_log / np.sqrt(n_steps)

    shocks = rng.standard_normal(n_steps)
    log_returns = step_drift - 0.5 * step_vol**2 + step_vol * shocks
    log_path = np.concatenate(([0.0], np.cumsum(log_returns)))
    path = start_price * np.exp(log_path)  # length n_steps + 1, on step boundaries

    boundaries = path[::steps_per_bin]  # length n_bins + 1
    bin_open_mid = boundaries[:-1].copy()
    close_mid = float(boundaries[-1])

    # Within-bin VWAP: average the step-boundary prices spanning the bin.
    within = np.lib.stride_tricks.sliding_window_view(path, steps_per_bin + 1)[::steps_per_bin]
    bin_vwap = within.mean(axis=1)

    weights = _volume_profile(n_bins, midday_floor=midday_floor)
    if volume_noise > 0.0:
        noise = np.exp(rng.normal(0.0, volume_noise, size=n_bins) - 0.5 * volume_noise**2)
        weights = weights * noise
        weights = weights / weights.sum()
    bin_volume = weights * session_volume

    return Tape(
        bin_vwap=np.ascontiguousarray(bin_vwap),
        bin_volume=np.ascontiguousarray(bin_volume),
        bin_open_mid=np.ascontiguousarray(bin_open_mid),
        close_mid=close_mid,
        bin_minutes=session_minutes / n_bins,
    )
