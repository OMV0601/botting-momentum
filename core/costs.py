"""
Transaction cost model for low-priced US equities.

The single biggest reason penny-universe backtests lie is that they charge
themselves nothing to trade. In a $5-20 universe the quoted spread is
routinely 30-200bp, so a strategy turning over daily pays 1.5-10% a week in
spread alone before any alpha shows up.

We estimate the spread from the data itself rather than assuming a constant,
using Corwin & Schultz (2012) "A Simple Way to Estimate Bid-Ask Spreads from
Daily High and Low Prices", Journal of Finance 67(2). That estimator exploits
the fact that the high/low range over two days reflects both volatility and
the spread, while volatility scales with time and the spread does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 3 - 2*sqrt(2), appears throughout Corwin-Schultz
_K = 3 - 2 * np.sqrt(2)

# Regulatory fees. These are the pass-through costs a zero-commission retail
# broker still charges. Rates change; these are the 2024-2025 values.
SEC_FEE_PER_DOLLAR_SOLD = 27.80 / 1_000_000   # SEC Section 31, sells only
FINRA_TAF_PER_SHARE = 0.000166                # sells only
FINRA_TAF_MAX_PER_TRADE = 8.30


def corwin_schultz_spread(
    high: pd.DataFrame | pd.Series,
    low: pd.DataFrame | pd.Series,
    window: int = 21,
) -> pd.DataFrame | pd.Series:
    """Rolling estimate of the proportional bid-ask spread from daily H/L.

    Returns spread as a fraction of price (0.01 == 100bp). Negative two-day
    estimates are floored at zero before averaging, per the paper's
    recommendation for the rolling-window version.

    NOTE: this uses only data available at time t, so it is safe to use as a
    cost input inside the backtest without introducing lookahead.
    """
    h, l = np.log(high), np.log(low)

    # Single-day squared log range, summed over two consecutive days.
    hl_sq = (h - l) ** 2
    beta = hl_sq + hl_sq.shift(1)

    # Two-day high and low, elementwise across the panel.
    h2 = np.maximum(h, h.shift(1))
    l2 = np.minimum(l, l.shift(1))
    gamma = (h2 - l2) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    two_day = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Floor at zero: negative estimates are noise, not negative spreads.
    two_day = two_day.where(two_day > 0, 0.0)
    return two_day.rolling(window, min_periods=max(3, window // 3)).mean()


class CostModel:
    """Charges a round trip honestly.

    Parameters
    ----------
    spread_floor_bp / spread_cap_bp
        Sanity bounds on the estimated spread. The floor stops the estimator
        from ever handing us a free trade; the cap stops one garbage print
        from dominating a backtest.
    spread_capture
        Fraction of the quoted spread actually paid. 0.5 == we always cross
        half the spread (marketable limit at the midpoint is optimistic;
        1.0 == we always pay the full spread by crossing). Default 1.0 is
        deliberately pessimistic: we assume we take liquidity on both sides.
    impact_coef
        Square-root market impact: impact = coef * spread * sqrt(participation).
        At $1000 of capital this is ~0, which is a genuine small-account edge,
        but it is modelled so the engine stays honest if capital is raised.
    borrow_apr_default
        Annualized borrow cost for shorts. Low-priced names are frequently
        hard-to-borrow; 0.30 (30%/yr) is a mid-range assumption and is applied
        per calendar day held.
    """

    def __init__(
        self,
        spread_floor_bp: float = 5.0,
        spread_cap_bp: float = 400.0,
        spread_capture: float = 1.0,
        impact_coef: float = 0.1,
        borrow_apr_default: float = 0.30,
        commission_per_trade: float = 0.0,
        sec_fee_rate: float = SEC_FEE_PER_DOLLAR_SOLD,
    ):
        self.spread_floor = spread_floor_bp / 1e4
        self.spread_cap = spread_cap_bp / 1e4
        self.spread_capture = spread_capture
        self.impact_coef = impact_coef
        self.borrow_apr = borrow_apr_default
        self.commission = commission_per_trade
        self.sec_fee_rate = sec_fee_rate

    def effective_spread(self, est_spread: pd.DataFrame) -> pd.DataFrame:
        """Clamp the raw estimate into a defensible band."""
        return est_spread.clip(lower=self.spread_floor, upper=self.spread_cap)

    def trade_cost(
        self,
        notional_traded: pd.DataFrame,
        est_spread: pd.DataFrame,
        adv_notional: pd.DataFrame | None = None,
        price: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Total cost in dollars for a given dollar volume traded.

        notional_traded is always non-negative (absolute value of the change
        in position). Both entering and exiting are charged.
        """
        spread = self.effective_spread(est_spread)
        # Crossing the spread costs half of it per side, and notional_traded
        # already counts each side separately.
        cost = notional_traded * spread * 0.5 * self.spread_capture

        if adv_notional is not None:
            participation = (notional_traded / adv_notional.replace(0, np.nan)).fillna(0.0)
            participation = participation.clip(0, 1)
            cost = cost + notional_traded * self.impact_coef * spread * np.sqrt(participation)

        # Regulatory: SEC fee on sells. We do not know direction here, so we
        # charge it on half the notional (the sell half of a round trip).
        cost = cost + notional_traded * 0.5 * self.sec_fee_rate

        if price is not None:
            shares = (notional_traded / price.replace(0, np.nan)).fillna(0.0)
            taf = (shares * 0.5 * FINRA_TAF_PER_SHARE).clip(upper=FINRA_TAF_MAX_PER_TRADE)
            cost = cost + taf

        if self.commission:
            cost = cost + (notional_traded > 0).astype(float) * self.commission
        return cost

    def borrow_cost(
        self, short_notional: pd.DataFrame, days: float = 1.0
    ) -> pd.DataFrame:
        """Daily financing charge on short exposure."""
        return short_notional.abs() * self.borrow_apr * (days / 365.0)


# --------------------------------------------------------------------------
# liquidity-tiered spread model
# --------------------------------------------------------------------------
#
# Corwin-Schultz is estimated from the daily high-low range, which for a
# liquid name is dominated by VOLATILITY rather than by the spread. Measured
# on the S&P 500 panel it returns a median of 50bp; the true quoted spread on
# those names is roughly 1-3bp. Using it there would overcharge by ~20x and
# manufacture a false negative -- the mirror image of the zero-cost backtest.
#
# So for liquid universes the spread is modelled from two things that actually
# bound it:
#
#   tick floor      the minimum tick is $0.01, so a stock cannot quote tighter
#                   than 1c/price. On a $200 name that is 0.5bp; on a $5 name
#                   it is 20bp. This alone explains most of why low-priced
#                   stocks are expensive to trade, and it is a hard constraint,
#                   not an estimate.
#
#   liquidity tier  anchored to published effective-spread values by dollar
#                   volume. Wide brackets on purpose: the point is to be in the
#                   right order of magnitude, not falsely precise.
#
# The result is capped by the Corwin-Schultz estimate where that is TIGHTER,
# so genuinely illiquid names still get their measured (wider) spread.

LIQUIDITY_TIERS_BP = [
    (100e6, 3.0),    # ADV > $100M   mega/large cap
    (10e6, 8.0),     # $10-100M      mid cap
    (1e6, 25.0),     # $1-10M        small cap
    (0.0, 90.0),     # < $1M         micro / illiquid
]


def tiered_spread(price, adv_notional, cs_estimate=None):
    """Spread as a fraction of price, from tick size and liquidity tier.

    Returns max(tick_floor, tier) and, when a Corwin-Schultz estimate is
    supplied, takes the WIDER of that and the tier value only for names the
    tier model treats as illiquid -- so the measured estimate can widen a
    cheap-looking name but cannot make a mega cap cost 50bp.
    """
    tick_floor = 0.01 / price.replace(0, np.nan)

    tier = pd.DataFrame(np.nan, index=adv_notional.index, columns=adv_notional.columns)
    for threshold, bp in LIQUIDITY_TIERS_BP:
        tier = tier.where(tier.notna(), np.nan)
        tier = tier.mask((adv_notional >= threshold) & tier.isna(), bp / 1e4)
    tier = tier.fillna(LIQUIDITY_TIERS_BP[-1][1] / 1e4)

    spread = tick_floor.combine(tier, np.maximum)

    if cs_estimate is not None:
        # Only let the measured estimate widen names the tier model already
        # considers illiquid (>= 25bp tier), where CS is actually reliable.
        illiquid = tier >= (25.0 / 1e4)
        spread = spread.where(~illiquid, spread.combine(cs_estimate, np.maximum))
    return spread


# --------------------------------------------------------------------------
# liquidity-tiered borrow cost
# --------------------------------------------------------------------------
#
# A flat 30% APR is right for hard-to-borrow microcaps and badly wrong for
# mega caps, where general collateral runs 25-50bp. Charged flat against a 50%
# short book it removes ~15%/yr, which is enough to turn a working long/short
# strategy into a losing one for reasons that have nothing to do with the
# signal. Same failure mode as Corwin-Schultz on liquid names: a cost model
# calibrated for one universe silently manufacturing a false negative in
# another.

BORROW_TIERS_APR = [
    (100e6, 0.005),   # ADV > $100M    general collateral, 50bp
    (10e6, 0.02),     # $10-100M       2%
    (1e6, 0.08),      # $1-10M         8%
    (0.0, 0.40),      # < $1M          hard to borrow, 40%
]


def tiered_borrow(adv_notional: pd.DataFrame) -> pd.DataFrame:
    """Annualized borrow rate per name, from dollar volume."""
    out = pd.DataFrame(np.nan, index=adv_notional.index, columns=adv_notional.columns)
    for threshold, apr in BORROW_TIERS_APR:
        out = out.mask((adv_notional >= threshold) & out.isna(), apr)
    return out.fillna(BORROW_TIERS_APR[-1][1])
