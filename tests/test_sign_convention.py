"""Positive always means it cost you. Both sides, everywhere."""

from __future__ import annotations

import pytest

from slippage_lab.benchmarks import slippage_bps
from slippage_lab.execution import Side


@pytest.mark.parametrize("benchmark", [10.0, 100.0, 1234.5])
def test_buy_above_benchmark_costs_you(benchmark: float) -> None:
    assert slippage_bps(benchmark * 1.001, benchmark, Side.BUY) == pytest.approx(10.0)
    assert slippage_bps(benchmark * 0.999, benchmark, Side.BUY) == pytest.approx(-10.0)


@pytest.mark.parametrize("benchmark", [10.0, 100.0, 1234.5])
def test_sell_below_benchmark_costs_you(benchmark: float) -> None:
    assert slippage_bps(benchmark * 0.999, benchmark, Side.SELL) == pytest.approx(10.0)
    assert slippage_bps(benchmark * 1.001, benchmark, Side.SELL) == pytest.approx(-10.0)


def test_the_two_sides_are_exact_mirrors() -> None:
    for price in (99.5, 100.0, 100.5, 101.25):
        buy = slippage_bps(price, 100.0, Side.BUY)
        sell = slippage_bps(price, 100.0, Side.SELL)
        assert buy == -sell


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_executing_at_the_benchmark_is_exactly_zero(side: Side) -> None:
    """Not approximately zero. A benchmark you matched cost you nothing."""
    assert slippage_bps(100.0, 100.0, side) == 0.0
    assert slippage_bps(37.125, 37.125, side) == 0.0


def test_the_formula_is_the_stated_one() -> None:
    # (execution - benchmark) / benchmark * 10_000 * side, and nothing else.
    assert slippage_bps(101.0, 100.0, Side.BUY) == pytest.approx(100.0)
    assert slippage_bps(101.0, 100.0, Side.SELL) == pytest.approx(-100.0)


def test_a_non_positive_benchmark_is_rejected() -> None:
    with pytest.raises(ValueError, match="benchmark price must be positive"):
        slippage_bps(100.0, 0.0, Side.BUY)
