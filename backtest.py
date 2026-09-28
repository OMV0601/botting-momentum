#!/usr/bin/env python3
"""
Backtest strategy.py on a price panel, starting from $5,000. Places no orders.

    python fetch_data.py --source alpaca --start 2016-01-01 --end <yesterday> \
        --n-tickers 4000 --out data/panel
    python backtest.py --panel data/panel

The engine (core/engine.py) is the assay project's: signal at the close of day
D, fill at the open of D+1, spread + fee costs charged on every trade, delisted
names settled at their last price (with a haircut if the price says the
company failed). Use the Alpaca source where possible -- it includes delisted
companies. Yahoo only has survivors, which biases results upward.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import strategy as S
from core.costs import CostModel
from core.data import Panel
from core.engine import Backtester
from core.metrics import max_drawdown, sharpe

CAPITAL = 5000.0
HORIZON, BLOCK, N_SIMS = 252, 21, 5000


def make_bt(panel: Panel) -> Backtester:
    return Backtester(
        panel, CostModel(), initial_capital=CAPITAL,
        min_price=S.MIN_PRICE, max_price=1e9,
        min_dollar_volume=S.MIN_DOLLAR_VOLUME,
        rebalance_band=0.005, spread_model="tiered",
        allow_short=False, max_gross=1.0,
    )


def one_year(r: np.ndarray, seed: int = 11) -> dict:
    """Chance of outcomes over the next 252 trading days, from resampled history."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(HORIZON / BLOCK))
    fin = np.empty(N_SIMS)
    for i in range(N_SIMS):
        s = rng.integers(0, len(r) - BLOCK, size=nb)
        fin[i] = np.prod(1 + np.concatenate([r[j:j + BLOCK] for j in s])[:HORIZON]) - 1
    return {"median": float(np.median(fin)), "p10": float(np.percentile(fin, 10)),
            "p90": float(np.percentile(fin, 90)),
            "p_hit_25": float((fin >= 0.25).mean()), "p_loss": float((fin < 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel")
    ap.add_argument("--out", default="results/backtest.json")
    a = ap.parse_args()

    panel = Panel.load(a.panel).trim_sparse_rows(0.3)
    print(f"[panel] {panel.source}  survivorship_free={panel.survivorship_free}  "
          f"{panel.close.index[0].date()} -> {panel.close.index[-1].date()}  "
          f"{panel.close.shape[1]} tickers")
    if not panel.survivorship_free:
        print("[panel] WARNING: survivor-only data -- results are biased HIGH")

    res = make_bt(panel).run(S.target_weights(panel.close, panel.volume, panel.open), S.NAME)
    start = res.n_positions[res.n_positions > 0].index[0]
    r = res.returns.loc[start:].dropna()
    eq = CAPITAL * (1 + r).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    yearly = pd.concat([pd.Series([CAPITAL], index=[eq.index[0]]),
                        eq.resample("YE").last()]).pct_change().dropna()
    live = res.n_positions.loc[start:]
    d = {
        "strategy": S.NAME, "description": S.DESCRIPTION,
        "source": panel.source, "start": str(start.date()), "end": str(eq.index[-1].date()),
        "start_capital": CAPITAL, "final_equity": float(eq.iloc[-1]),
        "cagr": float((eq.iloc[-1] / CAPITAL) ** (1 / yrs) - 1),
        "vol": float(r.std() * np.sqrt(252)), "sharpe": sharpe(r),
        "max_dd": max_drawdown(eq),
        "turnover_per_day": float(res.turnover.loc[start:].mean()),
        "cost_per_year": float(res.costs.loc[start:].sum() / res.equity.loc[start:].mean() / yrs),
        "avg_names": float(live[live > 0].mean()),
        "years": {str(k.year): float(v) for k, v in yearly.items()},
        "next_year": one_year(r.to_numpy()),
    }

    o = d["next_year"]
    print(f"\n{S.NAME}: {S.DESCRIPTION}")
    print(f"{d['start']} -> {d['end']}")
    print("=" * 60)
    print(f"  $5,000 grew to      ${d['final_equity']:>12,.0f}")
    print(f"  yearly return       {d['cagr']:>12.1%}")
    print(f"  volatility          {d['vol']:>12.1%}")
    print(f"  Sharpe              {d['sharpe']:>12.2f}")
    print(f"  worst drop          {d['max_dd']:>12.1%}")
    print(f"  names held (avg)    {d['avg_names']:>12.0f}")
    print(f"  turnover / day      {d['turnover_per_day']:>12.1%}")
    print(f"  trading cost / yr   {d['cost_per_year']:>12.1%}")
    print("-" * 60)
    for y, v in d["years"].items():
        print(f"  {y}  {v:>+8.1%}")
    print("-" * 60)
    print(f"  next 12 months (resampled): median {o['median']:+.1%}, "
          f"bad case {o['p10']:+.1%}, good case {o['p90']:+.1%}")
    print(f"  chance of >= +25%: {o['p_hit_25']:.0%}    chance of a loss: {o['p_loss']:.0%}")
    print("=" * 60)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2, default=float))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
