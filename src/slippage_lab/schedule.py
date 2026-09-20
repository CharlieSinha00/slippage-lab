"""Child-order schedules: how a parent order is sliced across bins.

Every schedule returns integer quantities that sum *exactly* to the parent
quantity. Shares are not allowed to evaporate in a rounding step; a schedule
that silently loses 37 shares will silently corrupt every cost number
downstream, and the loss will look like slippage.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: A schedule builder: (parent quantity, market volume per bin, lot size).
Builder = Callable[[int, FloatArray, int], IntArray]


def apportion(weights: FloatArray, total_quantity: int, *, lot: int = 1) -> IntArray:
    """Split ``total_quantity`` across bins in proportion to ``weights``.

    Uses largest-remainder apportionment on whole lots. Flooring alone loses
    shares and rounding alone can overshoot the parent; largest remainder is
    the standard fix and is deterministic (ties break toward the earlier bin,
    because trading earlier is the conservative choice when nothing else
    distinguishes two bins).

    If ``total_quantity`` is not a whole number of lots, the odd lot is placed
    in the first bin with non-zero weight rather than dropped.
    """
    if total_quantity < 0:
        raise ValueError("total_quantity must be non-negative")
    if lot < 1:
        raise ValueError("lot must be at least 1")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    weight_sum = weights.sum()
    if weight_sum <= 0.0:
        raise ValueError("weights must not sum to zero")

    quantities = np.zeros(weights.size, dtype=np.int64)
    if total_quantity == 0:
        return quantities

    n_lots, odd_shares = divmod(total_quantity, lot)
    share = weights / weight_sum
    exact = share * n_lots
    base = np.floor(exact).astype(np.int64)
    remaining = int(n_lots - base.sum())
    if remaining > 0:
        # Sort by descending fractional part; np.argsort is stable, so equal
        # fractions keep ascending bin order and the result is reproducible.
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:remaining]] += 1

    quantities = base * lot
    if odd_shares:
        quantities[int(np.flatnonzero(weights > 0.0)[0])] += odd_shares

    if int(quantities.sum()) != total_quantity:  # pragma: no cover - defensive
        raise AssertionError("apportionment lost shares")
    return quantities


def twap(parent_quantity: int, n_bins: int, *, lot: int = 1) -> IntArray:
    """Equal quantity per bin. Ignores market volume, which is the whole idea."""
    return apportion(np.ones(n_bins, dtype=np.float64), parent_quantity, lot=lot)


def pov(parent_quantity: int, bin_volume: FloatArray, *, lot: int = 1) -> IntArray:
    """Quantity proportional to each bin's market volume.

    This is the schedule that tracks interval VWAP by construction: if you
    trade a constant fraction of every print, your average price is the market's
    average price plus whatever your own trading costs. Remember that when
    reading its VWAP slippage -- it is close to a tautology, not a triumph.
    """
    return apportion(np.asarray(bin_volume, dtype=np.float64), parent_quantity, lot=lot)


def front_loaded(
    parent_quantity: int, n_bins: int, *, decay: float = 0.55, lot: int = 1
) -> IntArray:
    """Geometric decay: most of the order early, a thin tail afterwards.

    ``decay`` is the ratio between consecutive bins, so bin *i* gets weight
    ``decay ** i``. Lower decay means more urgency and more market impact.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly between 0 and 1")
    weights = decay ** np.arange(n_bins, dtype=np.float64)
    return apportion(weights, parent_quantity, lot=lot)


SCHEDULES: dict[str, Builder] = {
    "twap": lambda quantity, volume, lot: twap(quantity, volume.size, lot=lot),
    "pov": lambda quantity, volume, lot: pov(quantity, volume, lot=lot),
    "front_loaded": lambda quantity, volume, lot: front_loaded(quantity, volume.size, lot=lot),
}
