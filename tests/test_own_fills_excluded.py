"""The headline correctness property: interval VWAP must not contain our own fills.

Benchmarking a fill against a VWAP that includes that fill is benchmarking
against yourself. The error is negligible at 1% participation and enormous at
40%, which is to say it is largest exactly when someone is leaning on the
number. These tests construct a case where the difference is unmissable and
pin the library to the correct side of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from slippage_lab.benchmarks import (
    interval_vwap,
    interval_vwap_including_own_fills,
    slippage_bps,
)
from slippage_lab.execution import ParentOrder, Side, execute
from slippage_lab.report import measure
from slippage_lab.tape import Tape

# Two bins, equal market volume, a ten-percent price gap between them.
MARKET_PRICE = np.array([100.0, 110.0])
MARKET_VOLUME = np.array([10_000.0, 10_000.0])


def two_bin_tape() -> Tape:
    return Tape(
        bin_vwap=MARKET_PRICE.copy(),
        bin_volume=MARKET_VOLUME.copy(),
        bin_open_mid=np.array([100.0, 110.0]),
        close_mid=110.0,
        bin_minutes=1.0,
    )


def test_interval_vwap_is_the_market_average_and_nothing_else() -> None:
    assert interval_vwap(MARKET_PRICE, MARKET_VOLUME) == pytest.approx(105.0)


def test_including_own_fills_drags_the_benchmark_toward_your_own_prints() -> None:
    own_price = np.array([100.0, 110.0])
    own_quantity = np.array([9_000.0, 100.0])
    contaminated = interval_vwap_including_own_fills(
        MARKET_PRICE, MARKET_VOLUME, own_price, own_quantity
    )
    # (100*10000 + 110*10000 + 100*9000 + 110*100) / (20000 + 9100)
    assert contaminated == pytest.approx(3_011_000.0 / 29_100.0)
    assert contaminated == pytest.approx(103.4708, abs=1e-4)
    assert contaminated < interval_vwap(MARKET_PRICE, MARKET_VOLUME)


def test_the_contamination_grows_with_participation() -> None:
    """Negligible for a small order, ruinous for a large one -- which is the trap.

    The error is quoted in basis points of the benchmark, because that is the
    unit the mistake eventually gets reported in.
    """
    errors_bps = []
    for own in (100.0, 1_000.0, 5_000.0, 9_000.0):
        contaminated = interval_vwap_including_own_fills(
            MARKET_PRICE, MARKET_VOLUME, np.array([100.0, 110.0]), np.array([own, 0.0])
        )
        errors_bps.append(abs(contaminated - 105.0) / 105.0 * 10_000.0)

    assert errors_bps == sorted(errors_bps), "more of your own volume must skew it further"
    assert errors_bps[0] < 5.0, "0.5% of the interval's volume should barely register"
    assert errors_bps[-1] > 100.0, "31% of the interval's volume moves it by over a percent"


def test_the_reported_slippage_uses_the_uncontaminated_benchmark() -> None:
    """The whole pipeline, not just the helper: what does the report actually score against?"""
    tape = two_bin_tape()
    order = ParentOrder(side=Side.BUY, quantity=9_100)
    execution = execute(
        tape,
        np.array([9_000, 100], dtype=np.int64),
        order,
        impact_coef=0.0,
        half_spread_bps=0.0,
        max_participation=0.95,
    )
    assert execution.filled_quantity == 9_100
    result = measure("front_loaded", execution, tape)

    average = execution.average_fill_price
    correct = slippage_bps(average, 105.0, Side.BUY)
    contaminated = slippage_bps(average, 3_011_000.0 / 29_100.0, Side.BUY)

    assert result.vwap_slippage_bps == pytest.approx(correct)
    assert result.vwap_slippage_bps != pytest.approx(contaminated)
    # Getting this wrong would flatter the order by well over a hundred bps.
    assert abs(contaminated - correct) > 100.0
    assert result.naive_vwap_slippage_bps == pytest.approx(contaminated)


def test_the_error_is_signed_against_you_when_you_buy_the_cheap_bin() -> None:
    """Contamination is not noise: it has a direction, and the direction is flattering."""
    tape = two_bin_tape()
    order = ParentOrder(side=Side.BUY, quantity=9_100)
    execution = execute(
        tape,
        np.array([9_000, 100], dtype=np.int64),
        order,
        impact_coef=0.0,
        half_spread_bps=0.0,
        max_participation=0.95,
    )
    result = measure("front_loaded", execution, tape)
    # Our prints sit in the cheap bin, so they pull the benchmark down, so a buy
    # looks worse than it was against the contaminated number, not better.
    assert result.naive_vwap_slippage_bps > result.vwap_slippage_bps


def test_interval_vwap_cannot_be_handed_own_fills_at_all() -> None:
    """The signature is the safeguard: there is no argument through which fills could leak."""
    with pytest.raises(TypeError):
        interval_vwap(MARKET_PRICE, MARKET_VOLUME, np.array([1.0]))  # type: ignore[call-arg]
