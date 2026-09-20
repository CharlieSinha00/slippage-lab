"""Command line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from slippage_lab.execution import Side
from slippage_lab.report import Scenario, render, run_scenario
from slippage_lab.tape import REGIMES


def build_parser() -> argparse.ArgumentParser:
    defaults = Scenario()
    parser = argparse.ArgumentParser(
        prog="slippage-lab",
        description=(
            "Execute one parent order three ways against a synthetic tape and compare "
            "interval VWAP against implementation shortfall. All data is generated."
        ),
    )
    parser.add_argument("--regime", choices=sorted(REGIMES), default=defaults.regime)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--side", choices=["buy", "sell"], default=defaults.side.name.lower())
    parser.add_argument("--quantity", type=int, default=defaults.quantity)
    parser.add_argument("--bins", type=int, default=defaults.n_bins)
    parser.add_argument(
        "--delay-bins",
        type=int,
        default=defaults.delay_bins,
        help="bins between the decision and the first child order",
    )
    parser.add_argument(
        "--limit-bps",
        type=float,
        default=defaults.limit_bps,
        help="parent limit price, in bps through the arrival mid",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="work the order with no limit price (every schedule then completes)",
    )
    parser.add_argument("--lot", type=int, default=defaults.lot)
    parser.add_argument("--session-volume", type=float, default=defaults.session_volume)
    parser.add_argument("--impact-coef", type=float, default=defaults.impact_coef)
    parser.add_argument("--half-spread-bps", type=float, default=defaults.half_spread_bps)
    parser.add_argument("--max-participation", type=float, default=defaults.max_participation)
    parser.add_argument("--vol-bps", type=float, default=defaults.vol_bps)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = Scenario(
        regime=args.regime,
        seed=args.seed,
        n_bins=args.bins,
        side=Side.BUY if args.side == "buy" else Side.SELL,
        quantity=args.quantity,
        session_volume=args.session_volume,
        delay_bins=args.delay_bins,
        limit_bps=None if args.no_limit else args.limit_bps,
        lot=args.lot,
        impact_coef=args.impact_coef,
        half_spread_bps=args.half_spread_bps,
        max_participation=args.max_participation,
        vol_bps=args.vol_bps,
    )
    print(render(run_scenario(scenario)), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
