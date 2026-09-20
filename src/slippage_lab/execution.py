"""Turning a schedule into fills against the tape.

The fill model is deliberately thin: a half-spread you always pay, plus a
square-root impact term in the bin's participation rate. Square root because
that is the shape the empirical impact literature keeps finding, and because
concavity is what makes an aggressive schedule expensive without making it
absurd.

Two frictions decide whether the parent order actually completes, and
completion is where the two benchmarks in this package part company:

* a participation cap, because no sane algo takes an unbounded share of a bin;
* an optional limit price on the parent, because a limit is the simplest
  realistic reason an order ends the day unfinished.

A slice that cannot be filled in its own bin is carried into the next one. That
is what a real algo does; abandoning the slice would flatter the schedules that
ask for too much too early.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from slippage_lab.tape import Tape

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class Side(IntEnum):
    """+1 for a buy, -1 for a sell. Defined once; used everywhere.

    Every cost in this package is signed so that *positive means it cost you*.
    For a buy that means paying above the benchmark; for a sell it means
    receiving below it. Multiplying a raw price difference by ``side`` is the
    only place that convention is applied.
    """

    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class ParentOrder:
    """The instruction a trader is given, before any slicing."""

    side: Side
    quantity: int
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("parent quantity must be positive")
        if self.limit_price is not None and self.limit_price <= 0.0:
            raise ValueError("limit price must be positive")

    def within_limit(self, price: float) -> bool:
        """True if ``price`` is acceptable: at or below a buy limit, at or above a sell limit."""
        if self.limit_price is None:
            return True
        return int(self.side) * (price - self.limit_price) <= 0.0


@dataclass(frozen=True)
class Execution:
    """What actually happened, bin by bin, over the whole tape."""

    order: ParentOrder
    quantity: IntArray
    """Shares filled in each tape bin (zero where nothing traded)."""

    price: FloatArray
    """Average fill price in each tape bin; NaN where nothing traded."""

    start_bin: int
    """First bin in which child orders were live, i.e. decision time plus delay."""

    @property
    def filled_quantity(self) -> int:
        return int(self.quantity.sum())

    @property
    def unfilled_quantity(self) -> int:
        return int(self.order.quantity - self.filled_quantity)

    @property
    def fill_ratio(self) -> float:
        return self.filled_quantity / self.order.quantity

    @property
    def average_fill_price(self) -> float:
        """Quantity-weighted average fill price, or NaN if nothing filled."""
        filled = self.filled_quantity
        if filled == 0:
            return float("nan")
        traded = self.quantity > 0
        return float((self.price[traded] * self.quantity[traded]).sum() / filled)

    @property
    def traded_bins(self) -> IntArray:
        return np.flatnonzero(self.quantity > 0).astype(np.int64)

    def fill_span(self) -> tuple[int, int] | None:
        """Inclusive first/last bin in which the order traded, or None if it never traded.

        This span, not the whole session, is the interval that "interval VWAP"
        is measured over. See ``benchmarks.interval_vwap`` for why that choice
        is both standard and dangerous.
        """
        traded = self.traded_bins
        if traded.size == 0:
            return None
        return int(traded[0]), int(traded[-1])


def impact_bps(participation_rate: float, *, impact_coef: float, half_spread_bps: float) -> float:
    """Cost of trading a bin, in basis points, as a function of participation.

    ``impact_coef`` is quoted as the impact of taking 100% of a bin's volume,
    which makes it easy to sanity-check: at 25% participation you pay half of it.
    """
    if participation_rate < 0.0:
        raise ValueError("participation rate must be non-negative")
    return half_spread_bps + impact_coef * float(np.sqrt(participation_rate))


def execute(
    tape: Tape,
    target: IntArray,
    order: ParentOrder,
    *,
    start_bin: int = 0,
    impact_coef: float = 50.0,
    half_spread_bps: float = 1.0,
    max_participation: float = 0.25,
) -> Execution:
    """Run ``target`` (one quantity per *trading* bin) against ``tape``.

    ``target`` covers bins ``start_bin .. n_bins - 1``; the bins before that are
    the decision-to-release delay, during which nothing trades but the price
    keeps moving. That is the entire reason delay cost exists.
    """
    target = np.asarray(target, dtype=np.int64)
    if not 0 <= start_bin < tape.n_bins:
        raise ValueError("start_bin must be a valid bin index")
    if target.size != tape.n_bins - start_bin:
        raise ValueError(
            f"target has {target.size} bins but {tape.n_bins - start_bin} trading bins remain"
        )
    if int(target.sum()) != order.quantity:
        raise ValueError("schedule does not sum to the parent quantity")
    if not 0.0 < max_participation <= 1.0:
        raise ValueError("max_participation must lie in (0, 1]")

    quantity = np.zeros(tape.n_bins, dtype=np.int64)
    price = np.full(tape.n_bins, np.nan, dtype=np.float64)

    pending = 0
    for offset, slice_quantity in enumerate(target):
        bin_index = start_bin + offset
        pending += int(slice_quantity)
        if pending == 0:
            continue

        volume = float(tape.bin_volume[bin_index])
        wanted = min(pending, int(max_participation * volume))
        if wanted <= 0:
            continue

        rate = wanted / volume
        cost = impact_bps(rate, impact_coef=impact_coef, half_spread_bps=half_spread_bps)
        fill_price = float(tape.bin_vwap[bin_index]) * (1.0 + int(order.side) * cost / 10_000.0)
        if not order.within_limit(fill_price):
            continue  # the market is through our limit; the slice waits.

        quantity[bin_index] = wanted
        price[bin_index] = fill_price
        pending -= wanted

    return Execution(order=order, quantity=quantity, price=price, start_bin=start_bin)
