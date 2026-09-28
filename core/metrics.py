"""
Performance metrics, including the ones that punish you for searching.

A raw Sharpe ratio computed after trying 200 parameter sets is not a Sharpe
ratio, it is the maximum of 200 draws from a noise distribution. The Deflated
Sharpe Ratio (Bailey & Lopez de Prado 2014, "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality") asks
the only question that matters: given that I ran N trials, how surprised
should I be by the best one?

If DSR < 0.95 the strategy has not cleared the bar of "probably not luck".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf / periods
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def sortino(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=1) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    if yrs <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    mdd = abs(max_drawdown(equity))
    return float(cagr / mdd) if mdd > 1e-9 else float("nan")


def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0,
                         periods: int = TRADING_DAYS) -> float:
    """P(true Sharpe > benchmark), correcting for skew and fat tails."""
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return float("nan")
    sr = sharpe(r, periods=periods) / np.sqrt(periods)   # per-period
    sr_b = benchmark_sr / np.sqrt(periods)
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    z = (sr - sr_b) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Sharpe you would expect from the LUCKIEST of n_trials worthless strategies."""
    if n_trials < 2:
        return 0.0
    e = np.exp(1)
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * e))
    return float(np.sqrt(sr_variance) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_variance: float | None = None,
                    periods: int = TRADING_DAYS) -> float:
    """Probability the strategy has genuine skill, given n_trials were run.

    sr_variance is the variance of ANNUALIZED Sharpe across the trials. If you
    do not have it, a conservative default of 0.25 (sd 0.5) is used.
    """
    if sr_variance is None:
        sr_variance = 0.25
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe(returns, benchmark_sr=sr0, periods=periods)


def summarize(result, n_trials: int = 1, sr_variance: float | None = None) -> dict:
    eq, r = result.equity, result.returns
    wk = result.weekly_returns
    out = {
        "source": result.source,
        "survivorship_free": result.survivorship_free,
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": result.cagr,
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(eq),
        "vol_annual": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "psr_vs_zero": probabilistic_sharpe(r),
        "deflated_sharpe": deflated_sharpe(r, n_trials, sr_variance),
        "n_trials_assumed": n_trials,
        "weekly_mean": float(wk.mean()) if len(wk) else float("nan"),
        "weekly_median": float(wk.median()) if len(wk) else float("nan"),
        "pct_weeks_over_50pct": float((wk > 0.50).mean()) if len(wk) else float("nan"),
        "best_week": float(wk.max()) if len(wk) else float("nan"),
        "worst_week": float(wk.min()) if len(wk) else float("nan"),
        "hit_rate_daily": float((r > 0).mean()),
        "total_costs": float(result.costs.sum()),
        "cost_drag_annual": float(
            result.costs.sum() / result.equity.iloc[0]
            / max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        ),
        "avg_turnover_daily": float(result.turnover.mean()),
        "avg_positions": float(result.n_positions.mean()),
    }
    return out


def format_summary(s: dict, title: str = "") -> str:
    warn = "" if s["survivorship_free"] else (
        "\n  !! SURVIVORSHIP-BIASED DATA (" + s["source"] + "): returns are inflated,\n"
        "     treat every number below as an optimistic upper bound."
    )
    return f"""
{'=' * 62}
{title or s.get('name', 'strategy')}   [{s['start']} -> {s['end']}]{warn}
{'-' * 62}
  CAGR                {s['cagr']:>10.2%}      Sharpe        {s['sharpe']:>8.2f}
  Total return        {s['total_return']:>10.2%}      Sortino       {s['sortino']:>8.2f}
  Final equity        {s['final_equity']:>10,.2f}      Calmar        {s['calmar']:>8.2f}
  Max drawdown        {s['max_drawdown']:>10.2%}      Ann. vol      {s['vol_annual']:>8.2%}
{'-' * 62}
  Mean WEEKLY return  {s['weekly_mean']:>10.2%}      (target was +50.00%)
  Best week           {s['best_week']:>10.2%}
  Worst week          {s['worst_week']:>10.2%}
  Weeks above +50%    {s['pct_weeks_over_50pct']:>10.2%}
{'-' * 62}
  Deflated Sharpe     {s['deflated_sharpe']:>10.3f}      (need > 0.95 to believe it)
  PSR vs zero         {s['psr_vs_zero']:>10.3f}
  Trials assumed      {s['n_trials_assumed']:>10d}
{'-' * 62}
  Cost drag / yr      {s['cost_drag_annual']:>10.2%}      Avg positions {s['avg_positions']:>8.1f}
  Daily turnover      {s['avg_turnover_daily']:>10.2%}      Daily hit     {s['hit_rate_daily']:>8.2%}
{'=' * 62}"""
