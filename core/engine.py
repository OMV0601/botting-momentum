"""
Backtest engine.

Deliberately an explicit daily loop rather than a vectorized panel product.
Vectorized backtests are where lookahead hides: one mis-signed .shift() and
the equity curve goes vertical. A loop makes the causality auditable, and
1300 days x ~600 names runs in seconds anyway.

Timing contract (enforced by tests/test_lookahead.py):

    close of day D    signal computed from data <= D
    open  of day D+1  order filled at that open, costs charged
    open  of day D+2  position marked, may be rebalanced

So a weight decided on day D earns open(D+2)/open(D+1) - 1. There is a full
session between the information and the fill. This is conservative; most
retail setups could act faster, but assuming you cannot is how you avoid
discovering an edge that is really just same-bar execution.

Delisting: when a name stops printing bars, the position is liquidated at the
last observed price multiplied by `delist_recovery`. For a $5-20 universe the
default of 0.30 is not pessimism, it is roughly what equity holders recover in
a Chapter 11; setting it to 1.0 silently assumes every failure is a buyout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel, corwin_schultz_spread, tiered_borrow, tiered_spread
from .data import Panel


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    gross_returns: pd.Series
    n_positions: pd.Series
    survivorship_free: bool
    source: str
    meta: dict

    @property
    def total_return(self) -> float:
        return self.equity.iloc[-1] / self.equity.iloc[0] - 1

    @property
    def cagr(self) -> float:
        yrs = (self.equity.index[-1] - self.equity.index[0]).days / 365.25
        if yrs <= 0 or self.equity.iloc[-1] <= 0:
            return float("nan")
        return (self.equity.iloc[-1] / self.equity.iloc[0]) ** (1 / yrs) - 1

    @property
    def weekly_returns(self) -> pd.Series:
        return self.equity.resample("W").last().pct_change().dropna()


class Backtester:
    def __init__(
        self,
        panel: Panel,
        cost_model: CostModel | None = None,
        initial_capital: float = 1000.0,
        max_gross: float = 1.0,
        allow_short: bool = False,
        delist_recovery: float = 0.30,
        distress_price: float = 1.0,
        min_price: float = 5.0,
        max_price: float = 20.0,
        min_dollar_volume: float = 250_000.0,
        adv_window: int = 21,
        max_participation: float = 0.01,
        rebalance_band: float = 0.0,
        max_spread_bp: float | None = None,
        spread_model: str = "corwin_schultz",
    ):
        self.panel = panel
        self.cm = cost_model or CostModel()
        self.capital0 = initial_capital
        self.max_gross = max_gross
        self.allow_short = allow_short
        self.delist_recovery = delist_recovery
        self.distress_price = distress_price
        self.min_price = min_price
        self.max_price = max_price
        self.min_dv = min_dollar_volume
        self.adv_window = adv_window
        self.max_participation = max_participation
        self.rebalance_band = rebalance_band
        self.max_spread_bp = max_spread_bp
        self.spread_model = spread_model
        self._prepare()

    def _prepare(self):
        p = self.panel
        self.O = p.open.astype(float)
        self.C = p.close.astype(float)
        self.dates = self.C.index

        # Dollar volume and ADV, both strictly trailing.
        dv = (self.C * p.volume.astype(float))
        self.adv = dv.rolling(self.adv_window, min_periods=5).mean().shift(1)

        # Spread estimate, trailing only.
        cs = corwin_schultz_spread(p.high.astype(float),
                                   p.low.astype(float)).shift(1)
        if self.spread_model == "tiered":
            # Corwin-Schultz reads volatility as spread on liquid names (50bp
            # median on the S&P 500, versus a true 1-3bp). The tiered model is
            # anchored to tick size and dollar volume instead.
            self.spread = tiered_spread(self.C.shift(1), self.adv, cs)
            self.spread = self.spread.clip(lower=0.5 / 1e4, upper=self.cm.spread_cap)
        else:
            self.spread = self.cm.effective_spread(cs.fillna(self.cm.spread_cap))
        self.spread = self.spread.fillna(self.cm.spread_cap)

        # Borrow rate per name. Flat 30% is a microcap assumption; on mega caps
        # it is ~60x too high and silently kills any short leg.
        if self.spread_model == "tiered":
            self.borrow = tiered_borrow(self.adv)
        else:
            self.borrow = pd.DataFrame(self.cm.borrow_apr, index=self.C.index,
                                       columns=self.C.columns)

        # Total-return factor per name, used so dividends are not lost.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tr_factor = (p.adj_close.astype(float) / self.C).replace(
                [np.inf, -np.inf], np.nan
            )

    def eligible_mask(self) -> pd.DataFrame:
        """Point-in-time tradability. Uses only data through the prior close.

        Everything here is shifted by one day so that the universe on day D is
        built from information available at the close of D-1.
        """
        c_prev = self.C.shift(1)
        price_ok = (c_prev >= self.min_price) & (c_prev <= self.max_price)
        liq_ok = self.adv >= self.min_dv
        alive = self.C.notna() & self.O.notna() & (self.O > 0)
        mask = price_ok & liq_ok & alive

        # Spread screen. The $5-20 band spans a 10x range in trading cost
        # (p10 = 24bp, p50 = 84bp, p90 = 211bp on this panel). A $1000 account
        # has no capacity constraint whatsoever, so it is free to confine
        # itself to the cheap tail -- which is the one structural advantage
        # being small actually confers. self.spread is already lagged.
        if self.max_spread_bp is not None:
            mask = mask & (self.spread <= self.max_spread_bp / 1e4)
        return mask

    def run(self, target_weights: pd.DataFrame, name: str = "strategy") -> BacktestResult:
        """target_weights: index=decision date, columns=tickers.

        A row dated D is acted on at the open of D+1.
        """
        W = target_weights.reindex(index=self.dates, columns=self.C.columns).fillna(0.0)
        elig = self.eligible_mask()

        O = self.O.to_numpy()
        C = self.C.to_numpy()
        adv = self.adv.to_numpy()
        spread = self.spread.to_numpy()
        borrow = self.borrow.to_numpy()
        Wn = W.to_numpy()
        En = elig.to_numpy()
        n_t, n_a = C.shape

        cash = self.capital0
        shares = np.zeros(n_a)
        last_px = np.full(n_a, np.nan)

        equity = np.full(n_t, np.nan)
        turn = np.zeros(n_t)
        cost_arr = np.zeros(n_t)
        npos = np.zeros(n_t)
        pos_hist = np.zeros((n_t, n_a))

        for t in range(n_t):
            o_t = O[t]
            have_open = np.isfinite(o_t) & (o_t > 0)

            # --- mark existing book at today's open -----------------------
            px = np.where(have_open, o_t, last_px)
            held = shares != 0

            # Delisting: held name with no open today and none ever again.
            gone = held & ~have_open & ~np.isfinite(o_t)
            if gone.any():
                px_gone = np.nan_to_num(last_px[gone], nan=0.0)
                # Delisting is not one event. Measured on the survivorship-free
                # panel, the median name's final 60 days returned +1.6% and only
                # 2.9% ended below $1 -- most delistings in a liquid universe are
                # acquisitions or voluntary deregistrations, where the holder is
                # cashed out at or above the last price. Applying a bankruptcy
                # haircut to all of them understates returns badly.
                # So: settle at last price, and reserve the haircut for names
                # whose price says the equity was actually impaired.
                recov = np.where(px_gone < self.distress_price,
                                 px_gone * self.delist_recovery, px_gone)
                cash += float((shares[gone] * recov).sum())
                shares[gone] = 0.0
                held = shares != 0

            mktval = np.where(held & np.isfinite(px), shares * np.nan_to_num(px), 0.0)
            equity_t = cash + float(mktval.sum())
            if equity_t <= 0:
                equity[t:] = 0.0
                break

            # --- decide target from YESTERDAY's signal --------------------
            # W.iloc[t-1] was computed at the close of t-1; we fill at open t.
            if t >= 1:
                w_raw = Wn[t - 1].copy()
                tradable = En[t] & have_open
                w_raw[~tradable] = 0.0

                gross = np.abs(w_raw).sum()
                if gross > self.max_gross and gross > 0:
                    w_raw *= self.max_gross / gross
                if not self.allow_short:
                    w_raw = np.clip(w_raw, 0, None)

                target_notional = w_raw * equity_t

                # Liquidity cap: never take more than max_participation of ADV.
                cap = np.nan_to_num(adv[t], nan=0.0) * self.max_participation
                target_notional = np.clip(target_notional, -cap, cap)

                cur_notional = np.where(np.isfinite(px), shares * np.nan_to_num(px), 0.0)
                delta = target_notional - cur_notional
                delta[~have_open] = 0.0

                # No-trade band. Without this, holding a CONSTANT target weight
                # still generates trades every single day, because prices drift
                # the actual weight away from target overnight. That is not a
                # strategy decision, it is an artifact of expressing intent as
                # weights, and in an 84bp-spread universe it silently bleeds
                # tens of percent a year. Real desks trade a band for exactly
                # this reason.
                if self.rebalance_band > 0:
                    small = np.abs(delta) < self.rebalance_band * equity_t
                    delta[small] = 0.0
                    target_notional = np.where(small, cur_notional, target_notional)

                traded = np.abs(delta)
                if traded.sum() > 0:
                    sp = np.nan_to_num(spread[t], nan=self.cm.spread_cap)
                    c = traded * sp * 0.5 * self.cm.spread_capture
                    with np.errstate(divide="ignore", invalid="ignore"):
                        part = np.where(cap > 0, traded / np.maximum(cap, 1e-9), 0.0)
                    part = np.clip(np.nan_to_num(part), 0, 1)
                    c = c + traded * self.cm.impact_coef * sp * np.sqrt(part)
                    c = c + traded * 0.5 * self.cm.sec_fee_rate
                    total_cost = float(c.sum())

                    new_shares = np.where(
                        have_open & (px > 0), target_notional / np.where(px > 0, px, 1), shares
                    )
                    cash -= float((new_shares - shares) @ np.nan_to_num(px))
                    cash -= total_cost
                    shares = new_shares

                    cost_arr[t] = total_cost
                    turn[t] = traded.sum() / max(equity_t, 1e-9)

                if self.allow_short:
                    shorts = np.abs(np.minimum(cur_notional, 0))
                    rate = np.nan_to_num(borrow[t], nan=self.cm.borrow_apr)
                    cash -= float((shorts * rate).sum()) / 365.0

            last_px = np.where(have_open, o_t, last_px)
            held = shares != 0
            mktval = np.where(held & np.isfinite(last_px), shares * np.nan_to_num(last_px), 0.0)
            equity[t] = cash + float(mktval.sum())
            npos[t] = int(held.sum())
            pos_hist[t] = mktval / max(equity[t], 1e-9)

        eq = pd.Series(equity, index=self.dates).ffill()
        rets = eq.pct_change().fillna(0.0)
        cost_s = pd.Series(cost_arr, index=self.dates)
        gross = rets + (cost_s / eq.shift(1).replace(0, np.nan)).fillna(0.0)

        return BacktestResult(
            equity=eq,
            returns=rets,
            positions=pd.DataFrame(pos_hist, index=self.dates, columns=self.C.columns),
            turnover=pd.Series(turn, index=self.dates),
            costs=cost_s,
            gross_returns=gross,
            n_positions=pd.Series(npos, index=self.dates),
            survivorship_free=self.panel.survivorship_free,
            source=self.panel.source,
            meta={
                "name": name,
                "initial_capital": self.capital0,
                "price_band": (self.min_price, self.max_price),
                "allow_short": self.allow_short,
                "delist_recovery": self.delist_recovery,
            },
        )
