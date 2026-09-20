"""The three benchmarks, and the one function that turns a price gap into bps.

Arrival price, interval VWAP, close. The interesting one is interval VWAP, for
two reasons that are easy to get wrong and expensive to get wrong quietly:

1. It must be computed from *market* volume only. Benchmarking your fills
   against a VWAP that includes those fills is benchmarking yourself against
   yourself. The error is small when you are 1% of the volume and large when
   you are 30% -- which is to say, it is largest exactly when someone is
   relying on the number.
2. You choose the interval. That is the sense in which interval VWAP is
   gameable: an order that only trades when the price suits it gets a benchmark
   window drawn around the bins where it happened to do well.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slippage_lab.execution import Side
from slippage_lab.tape import Tape

FloatArray = NDArray[np.float64]


def slippage_bps(execution_price: float, benchmark: float, side: Side | int) -> float:
    """Slippage in basis points. Positive always means it cost you.

    A buy filled above the benchmark and a sell filled below it both return a
    positive number. This is the only place the sign convention is applied.
    """
    if benchmark <= 0.0:
        raise ValueError("benchmark price must be positive")
    return (execution_price - benchmark) / benchmark * 10_000.0 * int(side)


def arrival_price(tape: Tape) -> float:
    """Mid at the decision instant.

    Not the first fill price. Using the first fill makes delay cost
    structurally zero, and a decomposition whose first term cannot be non-zero
    is decoration rather than analysis.
    """
    return tape.decision_mid


def close_price(tape: Tape) -> float:
    """Mid at the end of the session; the mark for anything left unfilled."""
    return tape.close_mid


def interval_vwap(bin_vwap: FloatArray, bin_volume: FloatArray) -> float:
    """Volume-weighted average price over an interval, from market volume alone.

    Takes plain arrays rather than a ``Tape`` and an ``Execution`` on purpose:
    there is no argument you could pass that would let your own fills leak into
    the result. Correct by construction beats correct by discipline.
    """
    bin_vwap = np.asarray(bin_vwap, dtype=np.float64)
    bin_volume = np.asarray(bin_volume, dtype=np.float64)
    if bin_vwap.shape != bin_volume.shape:
        raise ValueError("prices and volumes must have the same shape")
    if bin_vwap.size == 0:
        return float("nan")
    total = bin_volume.sum()
    if total <= 0.0:
        raise ValueError("interval has no volume")
    return float((bin_vwap * bin_volume).sum() / total)


def interval_vwap_including_own_fills(
    bin_vwap: FloatArray,
    bin_volume: FloatArray,
    own_price: FloatArray,
    own_quantity: FloatArray,
) -> float:
    """The wrong number, computed deliberately.

    This is the VWAP you get from the printed tape, which contains your own
    prints. It exists so the report and the tests can quantify how far off it
    is rather than assert that it is bad. Never use it as a benchmark.
    """
    bin_vwap = np.asarray(bin_vwap, dtype=np.float64)
    bin_volume = np.asarray(bin_volume, dtype=np.float64)
    own_price = np.asarray(own_price, dtype=np.float64)
    own_quantity = np.asarray(own_quantity, dtype=np.float64)
    traded = own_quantity > 0.0
    notional = (bin_vwap * bin_volume).sum() + (own_price[traded] * own_quantity[traded]).sum()
    volume = bin_volume.sum() + own_quantity[traded].sum()
    if volume <= 0.0:
        raise ValueError("interval has no volume")
    return float(notional / volume)
