"""
PRICE MOMENTUM (12-1), long-only.

Source: Kakushadze & Serur, "151 Trading Strategies" (SSRN 3247865), Sec 3.1.

--------------------------------------------------------------------------
THE RULE
--------------------------------------------------------------------------
Every trading day, for every stock in the universe:

    momentum = price 21 trading days ago / price 252 trading days ago - 1

i.e. the last 12 months' return, skipping the most recent month (the most
recent month tends to reverse, so it is left out). Buy the top 10% of the
universe by that number, equal weight.

The paper holds each pick for a month. Rebalancing everything on one fixed day
a month would make results depend on which day was picked, so the book is run
as 21 overlapping sleeves instead: each sleeve is 1/21 of capital, one sleeve
refreshes every trading day, and each holds its picks for 21 trading days
(Jegadeesh-Titman). Net effect: today's target is the average of the last 21
daily top-10% lists, so a name's weight builds up over the days it keeps
qualifying and fades out after it stops.

--------------------------------------------------------------------------
UNIVERSE (same as the assay bot, so the backtests are comparable)
--------------------------------------------------------------------------
    price >= $3, 21-day average dollar volume >= $5M, top 490 by liquidity.
    All inputs lagged one day.

--------------------------------------------------------------------------
BACKTEST (2017-2026, Alpaca survivorship-free data, costs included)
--------------------------------------------------------------------------
    CAGR 13.0%   vol 31.9%   Sharpe 0.54   max drawdown -52.3%
    Typically ~125 names. Turnover ~3% of the book per day.

Long-only, unlevered, no stop-losses. Positions leave when they stop ranking
in the top 10%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "momentum"

# ============================ PARAMETERS ==================================
LOOKBACK = 252          # 12 months
SKIP = 21               # skip the most recent month
TOP_FRACTION = 0.10     # buy the top decile
HOLD_DAYS = 21          # one month, run as 21 overlapping daily sleeves

MIN_PRICE = 3.0
MIN_DOLLAR_VOLUME = 5e6
UNIVERSE_SIZE = 490     # most liquid N, ranked point-in-time
# ==========================================================================

DESCRIPTION = (f"12-1 price momentum, top {TOP_FRACTION:.0%} equal weight, "
               f"{HOLD_DAYS} overlapping monthly sleeves, long-only, unlevered")
LONGEST_LOOKBACK = LOOKBACK


def eligible(close, volume, open_=None):
    """Point-in-time tradable set. Everything is lagged one day."""
    adv = (close * volume).rolling(21, min_periods=5).mean().shift(1)
    ok = (close.shift(1) >= MIN_PRICE) & (adv >= MIN_DOLLAR_VOLUME) & close.notna()
    if open_ is not None:
        ok = ok & open_.notna()
    return ok & (adv.rank(axis=1, ascending=False, na_option="keep") <= UNIVERSE_SIZE)


def score(close, elig) -> pd.DataFrame:
    """12-1 momentum. Higher is better. NaN where not eligible."""
    return (close.shift(SKIP) / close.shift(LOOKBACK) - 1).where(elig)


def daily_picks(close, volume, open_) -> pd.DataFrame:
    """One day's top-decile book, equal weight, summing to 1 (0 if empty)."""
    elig = eligible(close, volume, open_)
    s = score(close, elig)
    n = (s.notna().sum(axis=1) * TOP_FRACTION).round().clip(lower=1)
    # method="first" breaks ties deterministically so exactly n names are picked.
    rank = s.rank(axis=1, ascending=False, method="first")
    pick = rank.le(n, axis=0) & s.notna()
    w = pick.astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def target_weights(close, volume, open_) -> pd.DataFrame:
    """Full weight history with the 21 overlapping sleeves applied."""
    w = daily_picks(close, volume, open_)
    out = None
    for phase in range(HOLD_DAYS):
        leg = w.iloc[phase::HOLD_DAYS].reindex(w.index).ffill().fillna(0.0) / HOLD_DAYS
        out = leg if out is None else out + leg
    return out


def todays_target(close, volume, open_) -> pd.Series:
    """The portfolio to hold right now: ticker -> weight, summing to <= 1.

    Names picked by an older sleeve that are no longer tradable today are
    dropped and their share left in cash -- the same thing the backtest engine
    does -- rather than spread over the rest.
    """
    w = target_weights(close, volume, open_).iloc[-1]
    ok = eligible(close, volume, open_).iloc[-1]
    w = w.where(ok, 0.0)
    return w[w > 1e-6].sort_values(ascending=False)
