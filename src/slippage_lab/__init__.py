"""slippage-lab: execution cost analysis on a synthetic tape.

All data in this package is generated. There is no market data here and no
network access anywhere in it.
"""

from slippage_lab.benchmarks import (
    arrival_price,
    close_price,
    interval_vwap,
    interval_vwap_including_own_fills,
    slippage_bps,
)
from slippage_lab.execution import Execution, ParentOrder, Side, execute
from slippage_lab.report import Comparison, Scenario, ScheduleResult, measure, render, run_scenario
from slippage_lab.schedule import apportion, front_loaded, pov, twap
from slippage_lab.shortfall import Shortfall, implementation_shortfall
from slippage_lab.tape import Tape, build_tape

__all__ = [
    "Comparison",
    "Execution",
    "ParentOrder",
    "Scenario",
    "ScheduleResult",
    "Shortfall",
    "Side",
    "Tape",
    "apportion",
    "arrival_price",
    "build_tape",
    "close_price",
    "execute",
    "front_loaded",
    "implementation_shortfall",
    "interval_vwap",
    "interval_vwap_including_own_fills",
    "measure",
    "pov",
    "render",
    "run_scenario",
    "slippage_bps",
    "twap",
]
