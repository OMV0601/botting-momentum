"""Properties strategy.py must keep, whichever strategy this repo runs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import strategy as S  # noqa: E402


def panel(n_days=600, n_names=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"T{i}" for i in range(n_names)]
    drift = rng.normal(0.0004, 0.0008, n_names)          # some trend up, some down
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(drift, 0.02, (n_days, n_names)), 0)),
                         idx, cols)
    open_ = close.shift(1).fillna(close)
    volume = pd.DataFrame(1e6, idx, cols)
    return close, volume, open_


def test_weights_long_only_and_unlevered():
    c, v, o = panel()
    w = S.target_weights(c, v, o)
    assert (w >= -1e-12).all().all()
    assert (w.sum(axis=1) <= 1 + 1e-9).all()
    t = S.todays_target(c, v, o)
    assert not t.empty and (t > 0).all() and t.sum() <= 1 + 1e-9


def test_invests_once_warmed_up():
    c, v, o = panel()
    gross = S.target_weights(c, v, o).sum(axis=1)
    assert gross.iloc[S.LONGEST_LOOKBACK + 60:].min() > 0.99


def test_no_lookahead():
    """Changing the future must not change any past weight."""
    c, v, o = panel()
    cut = 450
    w_full = S.target_weights(c, v, o)
    c2 = c.copy()
    c2.iloc[cut + 1:] *= 3.0
    o2 = c2.shift(1).fillna(c2)
    w_mod = S.target_weights(c2, v, o2)
    pd.testing.assert_frame_equal(w_full.iloc[:cut + 1], w_mod.iloc[:cut + 1])


def test_todays_target_is_the_last_row():
    c, v, o = panel()
    t = S.todays_target(c, v, o)
    last = S.target_weights(c, v, o).iloc[-1]
    last = last[last > 1e-6]
    assert set(t.index) <= set(last.index)
    for sym, w in t.items():
        assert w == pytest.approx(last[sym])


def test_ineligible_names_never_held():
    c, v, o = panel()
    v = v.copy()
    v["T0"] = 1.0                                         # far below $5M/day
    assert S.todays_target(c, v, o).get("T0", 0.0) == 0.0


@pytest.mark.skipif(S.NAME != "momentum", reason="momentum-specific")
def test_momentum_buys_the_winners():
    c, v, o = panel()
    elig = S.eligible(c, v, o)
    today = S.daily_picks(c, v, o).iloc[-1]
    s = S.score(c, elig).iloc[-1].dropna()
    picked = today[today > 0].index
    n = int(round(len(s) * S.TOP_FRACTION))
    assert len(picked) == n
    assert set(picked) == set(s.sort_values(ascending=False).index[:n])


@pytest.mark.skipif(S.NAME != "momentum", reason="momentum-specific")
def test_momentum_is_average_of_last_hold_days():
    """21 sleeves == the average of the last 21 daily books, however the
    fetched history happens to be aligned. This is what makes the live
    target independent of how much history the daily run downloads."""
    c, v, o = panel()
    daily = S.daily_picks(c, v, o)
    w = S.target_weights(c, v, o).iloc[-1]
    pd.testing.assert_series_equal(w, daily.iloc[-S.HOLD_DAYS:].mean(), check_names=False)
    w_short = S.target_weights(c.iloc[5:], v.iloc[5:], o.iloc[5:]).iloc[-1]
    pd.testing.assert_series_equal(w, w_short, check_names=False)


@pytest.mark.skipif(S.NAME != "moving-average", reason="MA-specific")
def test_ma_holds_exactly_the_uptrends():
    c, v, o = panel()
    t = S.todays_target(c, v, o)
    up = S.in_uptrend(c).iloc[-1] & S.eligible(c, v, o).iloc[-1]
    assert set(t.index) == set(up[up].index)
    assert np.allclose(t.to_numpy(), 1 / up.sum())
