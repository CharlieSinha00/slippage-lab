# slippage-lab

Interval VWAP is a benchmark you can beat by accident and game on purpose.
Implementation shortfall is the one that pays the bill. `slippage-lab` builds a
synthetic tape, executes the same parent order three ways against it, and shows
the schedule that wins on VWAP losing on shortfall.

Everything in this repository is generated. There is no market data here, no
network access anywhere in the code, and nothing proprietary.

## Quickstart

```
git clone https://github.com/CharlieSinha00/slippage-lab && cd slippage-lab
uv run slippage-lab
```

`uv` fetches Python 3.12 and the dependencies on first run. To see the other
side of the argument:

```
uv run slippage-lab --no-limit            # every schedule completes
uv run slippage-lab --regime flat --seed 3
uv run slippage-lab --side sell --delay-bins 4
uv run pytest -q
```

## What one run looks like

```
slippage-lab  (synthetic data; nothing here is a real market)

  regime            trending  (expected drift +250 bps)
  seed              0
  order             BUY 400,000 shares
  session           26 bins x 15 min, 8,000,000 shares of market volume
  order / session   5.0%
  decision delay    1 bin(s)
  limit price       100.7000  (+70 bps vs arrival)

  arrival mid       100.0000
  close mid         102.5024   (+250.2 bps vs arrival)

  Cost against each benchmark (positive = it cost you)

               fill % part %   avg px vwap slip bps IS total bps   delay   exec    opp  filled
  schedule
  twap           20.0    3.6 100.3373         +13.5       +206.9    +4.2   +2.5 +200.2   80000
  pov            29.4    5.3 100.3283         +12.6       +186.2    +6.3   +3.4 +176.5  117800
  front_loaded   95.0   17.2 100.3818         +17.9        +48.8   +20.2  +16.1  +12.6  379900

  Interval VWAP with and without the order's own prints in it

               interval VWAP ...incl own fills slip bps (correct) slip bps (contaminated) error bps
  schedule
  twap              100.2025          100.2072              +13.5                   +13.0      -0.5
  pov               100.2025          100.2088              +12.6                   +11.9      -0.6
  front_loaded      100.2025          100.2288              +17.9                   +15.3      -2.6

  Interval VWAP ranks first:            pov  (+12.6 bps)
  Implementation shortfall ranks first: front_loaded  (+48.8 bps)

  The benchmarks disagree. pov beats front_loaded by 5.3 bps on interval VWAP
  and loses to it by 137.4 bps on shortfall. pov left 282,200 shares (71%) unfilled;
  interval VWAP never charged for them and shortfall charged +176.5 bps.
```

Read the two benchmark columns against each other. `pov` is the best schedule
in the session by interval VWAP, 5.3 bps ahead of `front_loaded` -- a gap large
enough to decide which broker keeps the flow. It is also 137 bps worse by
implementation shortfall, because it bought 29% of the order and then watched
the stock finish 250 bps higher without it. Interval VWAP scores the prices you
got. It has no opinion at all about the shares you did not get, and in a
trending market the shares you did not get are the entire story.

The divergence is not a property of this seed. Across seeds 0-23 on the default
scenario the two benchmarks pick different winners 22 times; `front_loaded` is
the shortfall winner in all 22. The two exceptions (seeds 7 and 23) are quiet
sessions where the limit barely binds, every schedule fills, and -- see below --
the benchmarks then *cannot* disagree.

## When the two benchmarks must agree

With every schedule fully filled over a common benchmark window, interval VWAP
slippage and shortfall are both increasing affine functions of the average fill
price. They are then guaranteed to rank schedules identically; no construction
can make them disagree. That is worth saying plainly, because it locates the
disagreement precisely: it lives in **completion** and in **choice of interval**,
not in the price you paid.

`uv run slippage-lab --no-limit` removes the limit price so that everything
fills, and the report says so:

```
               fill % part %   avg px vwap slip bps IS total bps   delay   exec   opp  filled
  schedule
  twap          100.0    5.3 101.6360         +17.3       +163.6   +21.2 +142.4  +0.0  400000
  pov           100.0    5.3 101.5874         +12.6       +158.7   +21.2 +137.5  +0.0  400000
  front_loaded  100.0   10.4 100.4160         -23.2        +41.6   +21.2  +20.4  +0.0  400000
```

Here `front_loaded` wins both, and its interval VWAP slippage is *negative*: it
beat VWAP by 23 bps. It did that by finishing in the first 14 bins of the
session, which is also the window its benchmark was drawn over. Same order, same
tape, benchmark chosen by the algorithm's own behaviour. That is the second way
interval VWAP moves under you, and the reason the default scenario carries a
limit price: a limit is the simplest realistic instruction that makes completion
differ between schedules.

## Three constraints a reader might not expect

**Interval VWAP excludes the order's own fills.** Your prints are on the tape.
A VWAP computed from the printed tape includes them, so you are partly
benchmarking yourself against yourself, and the error grows with participation
-- it is largest exactly when someone is leaning on the number. `interval_vwap`
takes market prices and market volumes and nothing else, so there is no argument
through which a fill could leak in. The wrong number is available as
`interval_vwap_including_own_fills`, used only to report the size of the
mistake: in the run above it flatters `front_loaded` by 2.6 bps at 17%
participation, and `tests/test_own_fills_excluded.py` constructs a case where it
is worth over 140 bps.

**Arrival price is the mid at the decision timestamp, not the first fill.**
Benchmarking to the first fill makes delay cost structurally zero and turns the
decomposition into decoration. The scenario has a `--delay-bins` parameter for
exactly this reason; on the run above, one bin of delay costs `front_loaded`
20.2 bps before it has traded a single share.

**Slippage is positive when it cost you.** `side` is `+1` for a buy and `-1` for
a sell, defined once in `execution.Side` and applied in one place
(`benchmarks.slippage_bps`). A buy filled above the benchmark and a sell filled
below it both report a positive number. Both sides are tested, including a
mirror test that runs the whole pipeline twice and asserts every cost flips sign
and nothing else changes.

## The Perold decomposition

Total shortfall is split into delay, execution and opportunity cost, all in
basis points of the **parent** notional at arrival:

| term | what it charges for | marked between |
| --- | --- | --- |
| delay | drift between the decision and the first child order | arrival mid, market entry mid |
| execution | spread, impact and intra-window timing | market entry mid, average fill |
| opportunity | shares that never traded | arrival mid, close |

Two conventions worth stating. Delay is charged on the executed quantity only,
because the unexecuted quantity's move from arrival all the way to the close is
already charged as opportunity cost and charging it twice would double count.
The denominator is the parent notional, not the executed notional, because
dividing by what you managed to trade is how a 29% fill comes to look like a
good day.

`total_bps` is computed independently of the three parts so that
`tests/test_shortfall.py` can assert they sum to it across 270 parameter
combinations rather than assert an identity the code was written to satisfy.

## Layout

| file | what is in it |
| --- | --- |
| `tape.py` | GBM price path with configurable drift, U-shaped volume profile, binned, deterministic under a seed |
| `schedule.py` | `twap`, `pov`, `front_loaded`, and the largest-remainder apportionment that keeps them summing to the parent |
| `execution.py` | `Side`, `ParentOrder`, the square-root impact model, the participation cap, the limit price and the fill loop |
| `benchmarks.py` | arrival, interval VWAP, close, and the one signed bps function |
| `shortfall.py` | implementation shortfall and the Perold decomposition |
| `report.py` / `cli.py` | run three schedules over one tape and lay the answers side by side |

About 700 lines of code, 1,100 with the docstrings. It is meant to be read end to end.

## What the model does and does not claim

The tape is geometric Brownian motion with a parabolic volume curve. The fill
model is a half-spread plus a square-root function of participation. Both are
standard first approximations and neither is calibrated to anything: the point
is the relationship between the benchmarks, which does not depend on the
constants, not a forecast of what an order would cost.

The `trending` regime uses a +250 bps expected session drift against 60 bps of
session volatility. That drift-to-noise ratio is higher than a typical session.
It is deliberate: at realistic intraday volatility a single seed is mostly a
coin flip, and the structural effect would be buried under the draw. Lower it
with `--vol-bps` and the effect survives on fewer seeds, which is itself the
honest result.

## Licence

MIT. See `LICENSE`.
