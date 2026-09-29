"""
Tests for the live trading path: exits, the dead man's switch, summary mode.

These cover the two failures that stopped the bot trading for four sessions in
August 2026 -- exits that could never complete, and a silence that looked
exactly like success. Both are the kind of bug a backtest cannot catch, because
neither is about the strategy.

Nothing here touches the network. paper_trade.api is monkeypatched everywhere
an Alpaca call would happen, so a test that accidentally tried to trade would
fail rather than reach the broker.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import notify
import paper_trade as pt

# Captured at import, before any fixture can stub it out: the tests that cover
# the account pin itself need the real function, not the rd fixture's stand-in.
_REAL_ACCOUNT_PIN = pt.assert_expected_account


def plan_row(symbol="AAA", weight=0.0, delta=-500.0, price=np.nan,
             held_qty=10.0, act=True):
    return {"symbol": symbol, "weight": weight, "target_$": 0.0,
            "current_$": 500.0, "delta_$": delta, "price": price,
            "held_qty": held_qty, "act": act}


@pytest.fixture
def captured_orders(monkeypatch):
    """Collect every order that would be sent, and send none of them."""
    sent = []

    def fake_api(path, method="GET", body=None, base=pt.PAPER_BASE):
        assert path == "/v2/orders" and method == "POST", \
            f"unexpected broker call: {method} {path}"
        sent.append(body)
        return {"id": "test-order-id"}

    monkeypatch.setattr(pt, "api", fake_api)
    return sent


# --------------------------------------------------------------------------
# The stranded-exit bug
# --------------------------------------------------------------------------

def test_full_exit_without_a_price_still_sells(captured_orders):
    """A held name missing from the data feed must still be exitable.

    This is the bug that stranded positions permanently. A name absent from the
    day's Yahoo panel got price=NaN and was skipped on every subsequent run, so
    the strategy could never drop it -- the opposite of a design whose only exit
    mechanism is the signal ranking a name out of the book.
    """
    df = pd.DataFrame([plan_row(price=np.nan, held_qty=7.5)])
    sent, failed = pt.submit(df)

    assert sent == 1, "a full exit with no feed price must still be sent"
    assert not failed
    order = captured_orders[0]
    assert order["side"] == "sell"
    assert float(order["qty"]) == 7.5, "must sell the broker's share count"


def test_full_exit_without_a_price_sells_exactly_what_is_held(captured_orders):
    """Never guess the size of an unpriced exit -- use the broker's number."""
    df = pd.DataFrame([plan_row(price=np.nan, held_qty=3.14159, delta=-9999.0)])
    pt.submit(df)
    assert float(captured_orders[0]["qty"]) == pytest.approx(3.14159)


def test_buy_without_a_price_is_still_skipped(captured_orders):
    """Buys need a price to size. Without one there is nothing to compute."""
    df = pd.DataFrame([plan_row(weight=0.05, delta=+500.0, price=np.nan,
                                held_qty=0.0)])
    sent, failed = pt.submit(df)
    assert sent == 0 and not captured_orders


def test_partial_sell_without_a_price_is_skipped(captured_orders):
    """A trim is sized from dollars, so it genuinely needs the price."""
    df = pd.DataFrame([plan_row(weight=0.02, delta=-100.0, price=np.nan,
                                held_qty=10.0)])
    sent, _ = pt.submit(df)
    assert sent == 0 and not captured_orders


def test_exit_is_skipped_when_nothing_is_held(captured_orders):
    df = pd.DataFrame([plan_row(price=np.nan, held_qty=0.0)])
    sent, _ = pt.submit(df)
    assert sent == 0 and not captured_orders


def test_partial_sell_never_exceeds_shares_held(captured_orders):
    """Regression: overshooting a sell is what drew HTTP 403 on live exits.

    delta_$ is valued at the broker's live price while `price` is the feed's
    last close, so dividing one by the other can ask for more shares than exist.
    """
    df = pd.DataFrame([plan_row(weight=0.02, delta=-1000.0, price=1.0,
                                held_qty=4.0)])
    pt.submit(df)
    assert float(captured_orders[0]["qty"]) <= 4.0


def test_full_exit_with_a_price_still_uses_the_share_count(captured_orders):
    """Even when priced, a full exit sells shares held rather than an estimate."""
    df = pd.DataFrame([plan_row(weight=0.0, delta=-500.0, price=49.0,
                                held_qty=10.0)])
    pt.submit(df)
    assert float(captured_orders[0]["qty"]) == 10.0


# --------------------------------------------------------------------------
# Open-to-close P&L
# --------------------------------------------------------------------------

def test_session_pnl_measures_open_to_close(monkeypatch):
    """Read the day's result off Alpaca's curve, not off when the job ran."""
    monkeypatch.setattr(pt, "api", lambda *a, **k: {
        "timestamp": [1, 2, 3],
        "equity": [101_012.68, 101_000.00, 100_991.30],
    })
    s = pt.session_pnl()

    assert s["open_equity"] == 101_012.68
    assert s["close_equity"] == 100_991.30
    assert s["pnl"] == pytest.approx(-21.38)
    assert s["n_points"] == 3


def test_session_pnl_skips_null_marks(monkeypatch):
    """Alpaca pads the series with nulls outside trading."""
    monkeypatch.setattr(pt, "api", lambda *a, **k: {
        "timestamp": [1, 2, 3, 4],
        "equity": [None, 100.0, None, 110.0],
    })
    s = pt.session_pnl()

    assert s["open_equity"] == 100.0 and s["close_equity"] == 110.0
    assert s["pnl"] == pytest.approx(10.0)
    assert s["n_points"] == 2


def test_session_pnl_requests_regular_hours_only(monkeypatch):
    """After-hours drift is not the session. 09-02's summary ran 2.5h late."""
    seen = {}
    monkeypatch.setattr(pt, "api",
                        lambda path, *a, **k: seen.update(path=path) or
                        {"timestamp": [1, 2], "equity": [1.0, 2.0]})
    pt.session_pnl()

    assert "extended_hours=false" in seen["path"]
    assert "period=1D" in seen["path"]


def test_session_pnl_returns_empty_when_too_short(monkeypatch):
    """One mark is not a session -- report nothing rather than a fake zero."""
    monkeypatch.setattr(pt, "api", lambda *a, **k: {
        "timestamp": [1], "equity": [100.0]})
    assert pt.session_pnl() == {}


def test_session_pnl_returns_empty_when_alpaca_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("history endpoint down")

    monkeypatch.setattr(pt, "api", boom)
    assert pt.session_pnl() == {}


# --------------------------------------------------------------------------
# The coverage denominator
# --------------------------------------------------------------------------

def test_tradable_universe_drops_delisted_names(monkeypatch):
    """The universe file is survivorship-inclusive; the live fetch must not be.

    Roughly 1,906 of its 3,998 tickers are acquired or liquidated companies.
    Yahoo is survivor-only, so measured against the raw file coverage tops out
    near 52% and the 95% guard can never pass -- which is exactly what happened
    every run from 2026-08-25 onward.
    """
    monkeypatch.setattr(pt, "api", lambda *a, **k: [
        {"symbol": "AAPL", "tradable": True},
        {"symbol": "MSFT", "tradable": True},
        {"symbol": "HALT", "tradable": False},
    ])
    live, filtered = pt.tradable_universe(["AAPL", "MSFT", "XLNX", "WORK", "HALT"])

    assert filtered
    assert live == ["AAPL", "MSFT"], "delisted and untradable names must go"


def test_tradable_universe_preserves_order(monkeypatch):
    monkeypatch.setattr(pt, "api", lambda *a, **k: [
        {"symbol": s, "tradable": True} for s in ("C", "B", "A")])
    live, _ = pt.tradable_universe(["A", "B", "C"])
    assert live == ["A", "B", "C"]


def test_tradable_universe_fails_closed_when_alpaca_is_unreachable(monkeypatch):
    """A broker outage must not silently widen the universe back out.

    Returning the raw list means the coverage guard then refuses the run. That
    is the safe direction: a missed day is visible and self-correcting, while a
    truncated universe quietly changes which names the signal picks.
    """
    def boom(*a, **k):
        raise RuntimeError("assets endpoint down")

    monkeypatch.setattr(pt, "api", boom)
    live, filtered = pt.tradable_universe(["AAPL", "XLNX"])

    assert not filtered
    assert live == ["AAPL", "XLNX"], "unfiltered, so the guard refuses the run"


def test_tradable_universe_refuses_an_empty_result(monkeypatch):
    monkeypatch.setattr(pt, "api", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="empty universe"):
        pt.tradable_universe(["AAPL", "MSFT"])


def test_coverage_guard_still_refuses_a_truncated_fetch(monkeypatch):
    """The threshold is NOT lowered. Fixing the denominator must not open a door.

    A genuine throttling event -- half the live names missing -- still has to
    stop the run.
    """
    idx = pd.bdate_range("2026-08-24", periods=5)
    syms = [f"S{i}" for i in range(10)]
    close = pd.DataFrame(1.0, index=idx, columns=syms)
    close.loc[:, syms[5:]] = np.nan          # half the live universe missing
    panel = type("P", (), {"close": close})()

    monkeypatch.setenv("MIN_COVERAGE", "0.95")
    with pytest.raises(RuntimeError, match="below MIN_COVERAGE"):
        pt._assert_coverage(panel, syms)


def test_fetch_stops_retrying_once_past_the_floor(monkeypatch):
    """Don't chase 100%. A few names are simply absent from Yahoo any given day.

    The probe spent 7.5 of its 18 minutes on four retry rounds that recovered
    none of the last 15 symbols, while coverage sat unchanged at 99.3%. Past the
    floor the run is going to be allowed to trade, so more rounds buy nothing.
    """
    idx = pd.bdate_range("2026-08-24", periods=5)
    syms = [f"S{i}" for i in range(100)]
    close = pd.DataFrame(1.0, index=idx, columns=syms)
    close.loc[:, syms[98:]] = np.nan          # 98% covered, above a 95% floor

    calls = []

    def fake_download(missing, start, end, batch, pause, threads):
        calls.append(len(missing))
        return type("P", (), {"close": close, "open": close, "high": close,
                              "low": close, "volume": close,
                              "adj_close": close})()

    monkeypatch.setattr(pt, "_yahoo_download", fake_download)
    monkeypatch.setattr(pt, "_merge_panels", lambda base, extra: extra)
    monkeypatch.setenv("MIN_COVERAGE", "0.95")
    monkeypatch.setenv("YF_ATTEMPTS", "5")

    pt._fetch_yahoo_with_retry(syms, "2026-01-01", "2026-09-02")

    assert len(calls) == 1, "one round cleared the floor; it must not retry"


def test_fetch_keeps_retrying_below_the_floor(monkeypatch):
    """A genuinely truncated fetch must still use its retries."""
    idx = pd.bdate_range("2026-08-24", periods=5)
    syms = [f"S{i}" for i in range(100)]
    close = pd.DataFrame(1.0, index=idx, columns=syms)
    close.loc[:, syms[50:]] = np.nan          # 50%, far below the floor

    calls = []

    def fake_download(missing, start, end, batch, pause, threads):
        calls.append(len(missing))
        return type("P", (), {"close": close})()

    monkeypatch.setattr(pt, "_yahoo_download", fake_download)
    monkeypatch.setattr(pt, "_merge_panels", lambda base, extra: extra)
    monkeypatch.setattr(pt.time, "sleep", lambda s: None)
    monkeypatch.setenv("MIN_COVERAGE", "0.95")
    monkeypatch.setenv("YF_ATTEMPTS", "3")
    monkeypatch.setenv("YF_BACKOFF", "0")

    pt._fetch_yahoo_with_retry(syms, "2026-01-01", "2026-09-02")

    assert len(calls) == 3, "below the floor, every retry must be used"


def test_coverage_guard_passes_when_the_live_universe_is_complete(monkeypatch):
    idx = pd.bdate_range("2026-08-24", periods=5)
    syms = [f"S{i}" for i in range(10)]
    panel = type("P", (), {"close": pd.DataFrame(1.0, index=idx, columns=syms)})()

    monkeypatch.setenv("MIN_COVERAGE", "0.95")
    pt._assert_coverage(panel, syms)          # must not raise


# --------------------------------------------------------------------------
# The dead man's switch
# --------------------------------------------------------------------------

@pytest.fixture
def rd(monkeypatch, tmp_path):
    """run_daily with its state and journal redirected into a temp dir."""
    import run_daily

    monkeypatch.setattr(run_daily, "STATE", tmp_path / "last_run.json")
    monkeypatch.setattr(run_daily, "SUMMARY_STATE", tmp_path / "last_summary.json")
    monkeypatch.setattr(run_daily, "ALERT_STATE", tmp_path / "last_alert.json")
    monkeypatch.setattr(run_daily, "HISTORY", tmp_path / "history.csv")
    monkeypatch.setattr(run_daily, "JOURNAL", tmp_path / "journal.md")
    # Default to "Alpaca's session record is available". Tests that care about
    # the fallback override it. Without this, session_pnl() reaches api() and
    # exits on missing credentials.
    monkeypatch.setattr(run_daily.pt, "session_pnl",
                        lambda: {"open_equity": 100_000.0, "close_equity": 100_050.0,
                                 "pnl": 50.0, "open_ts": 0, "close_ts": 1,
                                 "n_points": 390})
    # Default to "the credentials resolve to the account we expect". Same
    # reason as session_pnl above: unpatched it reaches api() and exits on
    # missing credentials. The pin itself is covered directly further down,
    # against the real function.
    monkeypatch.setattr(run_daily.pt, "assert_expected_account",
                        lambda: "PA0TESTACCT1")
    return run_daily


def test_staleness_alert_fires_after_a_gap(rd, monkeypatch):
    rd.STATE.write_text(json.dumps({"date": "2026-08-25", "equity": 101000}))
    alerts = []
    monkeypatch.setattr(rd.notify, "failure_alert",
                        lambda stage, detail: alerts.append((stage, detail)))

    rd.check_staleness("2026-08-31", "market closed")

    assert len(alerts) == 1, "four trading days of silence must raise an alert"
    assert "2026-08-25" in alerts[0][1]


def test_staleness_alert_is_quiet_when_current(rd, monkeypatch):
    rd.STATE.write_text(json.dumps({"date": "2026-08-31"}))
    alerts = []
    monkeypatch.setattr(rd.notify, "failure_alert",
                        lambda stage, detail: alerts.append(detail))

    rd.check_staleness("2026-09-01", "market closed")

    assert not alerts, "one day is a normal gap, not an outage"


def test_staleness_alert_fires_once_per_day(rd, monkeypatch):
    """The schedule fires ~16 times a day; the alert must not."""
    rd.STATE.write_text(json.dumps({"date": "2026-08-25"}))
    alerts = []
    monkeypatch.setattr(rd.notify, "failure_alert",
                        lambda stage, detail: alerts.append(detail))

    for _ in range(5):
        rd.check_staleness("2026-08-31", "market closed")

    assert len(alerts) == 1


def test_staleness_is_silent_with_no_history(rd, monkeypatch):
    """A first-ever run has nothing to be stale against."""
    alerts = []
    monkeypatch.setattr(rd.notify, "failure_alert",
                        lambda stage, detail: alerts.append(detail))
    rd.check_staleness("2026-08-31", "market closed")
    assert not alerts


def test_trading_days_skips_weekends(rd):
    from datetime import date
    # Fri 2026-08-28 -> Mon 2026-08-31 is one trading day, not three.
    assert rd.trading_days_between(date(2026, 8, 28), date(2026, 8, 31)) == 1


# --------------------------------------------------------------------------
# Close-of-day summary must never trade
# --------------------------------------------------------------------------

def test_summary_only_places_no_orders(rd, monkeypatch):
    """--summary-only must not be able to reach the order path at all."""
    monkeypatch.setattr(rd.pt, "account", lambda: (101_130.0, 94_584.0, "ACTIVE"))
    monkeypatch.setattr(rd.pt, "positions", lambda: {"AAA": 300.0, "BBB": 250.0})

    def explode(*a, **k):
        raise AssertionError("summary mode must never build a plan or submit")

    monkeypatch.setattr(rd.pt, "build_plan", explode)
    monkeypatch.setattr(rd.pt, "submit", explode)
    monkeypatch.setattr(rd.notify, "close_summary", lambda **kw: True)

    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--summary-only"])
    assert rd.main() == 0


def test_second_timezone_fire_does_not_send_a_duplicate(rd, monkeypatch):
    """Two crons fire every weekday; only one email may go out.

    On 2026-09-02 both the EDT (20:10 UTC) and EST (21:10 UTC) fires ran and
    both emailed, ~40 minutes apart. The trading path absorbs its duplicate
    fire with a once-per-day gate; this one had none.
    """
    monkeypatch.setattr(rd.pt, "account", lambda: (101_021.72, 94_584.03, "ACTIVE"))
    monkeypatch.setattr(rd.pt, "positions", lambda: {"AAA": 300.0})
    calls = []
    monkeypatch.setattr(rd.notify, "close_summary",
                        lambda **kw: calls.append(kw) or True)

    rd.close_of_day_summary("2026-09-02")
    rd.close_of_day_summary("2026-09-02")     # the other timezone's fire

    assert len(calls) == 1, "the second fire must not send a second email"


def test_a_failed_send_leaves_the_day_open_to_retry(rd, monkeypatch):
    """Gate on a recorded send, not on the clock, so the extra fire is a retry.

    Same reasoning as the coverage refusal not consuming the trading day: a
    failure that blocks its own retry is worse than the duplicate it prevents.
    """
    monkeypatch.setattr(rd.pt, "account", lambda: (101_021.72, 94_584.03, "ACTIVE"))
    monkeypatch.setattr(rd.pt, "positions", lambda: {"AAA": 300.0})
    calls = []

    def failing(**kw):
        calls.append(kw)
        return False                          # Resend refused it

    monkeypatch.setattr(rd.notify, "close_summary", failing)
    rd.close_of_day_summary("2026-09-02")
    assert not rd.SUMMARY_STATE.exists(), "a failed send must not claim the day"

    monkeypatch.setattr(rd.notify, "close_summary",
                        lambda **kw: calls.append(kw) or True)
    rd.close_of_day_summary("2026-09-02")

    assert len(calls) == 2, "the later fire must retry after a failed send"
    assert rd.SUMMARY_STATE.exists()


def test_a_new_day_sends_again(rd, monkeypatch):
    monkeypatch.setattr(rd.pt, "account", lambda: (101_021.72, 94_584.03, "ACTIVE"))
    monkeypatch.setattr(rd.pt, "positions", lambda: {"AAA": 300.0})
    calls = []
    monkeypatch.setattr(rd.notify, "close_summary",
                        lambda **kw: calls.append(kw) or True)

    rd.close_of_day_summary("2026-09-02")
    rd.close_of_day_summary("2026-09-03")

    assert len(calls) == 2, "the gate is per-day, not permanent"


def test_summary_reports_the_book_and_the_day(rd, monkeypatch):
    rd.STATE.write_text(json.dumps({"date": "2026-08-25", "equity": 101_000.0,
                                    "deployed": 5000.0, "sent": 32,
                                    "rejected": 0}))
    monkeypatch.setattr(rd.pt, "account", lambda: (101_130.0, 94_584.0, "ACTIVE"))
    monkeypatch.setattr(rd.pt, "positions", lambda: {"AAA": 6000.0, "BBB": 546.0})

    got = {}
    monkeypatch.setattr(rd.notify, "close_summary",
                        lambda **kw: got.update(kw) or True)

    rd.close_of_day_summary("2026-09-02")

    assert got["position_value"] == pytest.approx(6546.0)
    assert got["n_positions"] == 2
    assert got["deployed"] == 5000.0
    # The baseline is the last completed run, not "this morning". Without the
    # date the email labelled a week of drift as one session's P&L.
    assert got["since_date"] == "2026-08-25"
    assert got["traded_today"] is False


def test_close_summary_leads_with_the_session_figure(monkeypatch):
    """The headline must be open-to-close, not "since the bot last ran".

    The old baseline was whenever the rebalance happened -- 14:24 UTC on
    2026-09-02, but hours later on a day GitHub's scheduler lags -- and the
    reading landed whenever the summary fired, 2.5 hours after the close that
    day. Two numbers measured over different spans cannot be compared.
    """
    sent = {}
    monkeypatch.setattr(notify, "send",
                        lambda subject, html: sent.update(subject=subject, html=html))

    notify.close_summary(equity=100_991.30, cash=95_943.75,
                         position_value=5047.55, n_positions=47,
                         deployed=5000.0, open_equity=100_969.94,
                         traded_today=True, since_date="2026-09-02",
                         session={"open_equity": 101_012.68,
                                  "close_equity": 100_991.30,
                                  "pnl": -21.38, "n_points": 390})

    assert "-21.38" in sent["subject"], "the subject must carry the real number"
    assert "market open to close" in sent["html"]
    assert "$101,012.68" in sent["html"] and "$100,991.30" in sent["html"]
    # +21.36 is the misleading since-last-run figure; it must not appear.
    assert "+21.36" not in sent["html"]


def test_close_summary_percentage_is_against_deployed_not_equity(monkeypatch):
    """-21.38 on a $5,000 book is -0.43%, not the -0.02% of a $101k account."""
    sent = {}
    monkeypatch.setattr(notify, "send",
                        lambda subject, html: sent.update(html=html))

    notify.close_summary(equity=100_991.30, cash=95_943.75,
                         position_value=5047.55, n_positions=47,
                         deployed=5000.0, traded_today=True,
                         session={"open_equity": 101_012.68,
                                  "close_equity": 100_991.30,
                                  "pnl": -21.38, "n_points": 390})

    assert "-0.43%" in sent["html"]


def test_close_summary_falls_back_loudly_without_the_session_record(monkeypatch):
    """If Alpaca's curve is unavailable, say so rather than quietly substituting.

    The fallback is the weaker measure. Presenting it as the day's result is
    exactly the false confidence this change exists to remove.
    """
    sent = {}
    monkeypatch.setattr(notify, "send",
                        lambda subject, html: sent.update(subject=subject, html=html))

    notify.close_summary(equity=101_021.72, cash=94_584.03,
                         position_value=6437.69, n_positions=40,
                         deployed=5000.0, open_equity=101_091.57,
                         traded_today=False, since_date="2026-08-25",
                         session=None)

    assert "session record was unavailable" in sent["html"]
    assert "not</b> an open-to-close figure" in sent["html"]
    assert "2026-08-25" in sent["html"]
    assert "since last run" in sent["subject"]


def test_close_summary_ignores_a_too_short_session(monkeypatch):
    """One equity mark is not a session; treat it as unavailable."""
    sent = {}
    monkeypatch.setattr(notify, "send",
                        lambda subject, html: sent.update(html=html))

    notify.close_summary(equity=101_021.72, cash=94_584.03,
                         position_value=6437.69, n_positions=40,
                         deployed=5000.0, open_equity=101_091.57,
                         traded_today=True,
                         session={"open_equity": 1.0, "close_equity": 1.0,
                                  "pnl": 0.0, "n_points": 1})

    assert "market open to close" not in sent["html"]
    assert "session record was unavailable" in sent["html"]


# --------------------------------------------------------------------------
# Liquidation guards -- the only thing here that places irreversible orders
# --------------------------------------------------------------------------

@pytest.fixture
def liq(monkeypatch, tmp_path):
    import liquidate

    monkeypatch.setattr(liquidate, "JOURNAL", tmp_path / "journal.md")
    monkeypatch.setattr(liquidate.pt, "account",
                        lambda: (101_130.0, 94_584.0, "ACTIVE"))
    monkeypatch.setattr(liquidate.pt, "positions",
                        lambda: {"AAA": 6000.0, "BBB": 546.0})
    monkeypatch.setattr(liquidate.pt, "position_qtys",
                        lambda: {"AAA": 35.0, "BBB": 1.35})
    return liquidate


def _api_recorder(monkeypatch, liq, is_open=True):
    calls = []

    def fake_api(path, method="GET", body=None, base=liq.pt.PAPER_BASE):
        calls.append((method, path))
        if path == "/v2/clock":
            return {"is_open": is_open, "next_open": "2026-09-02T09:30:00-04:00"}
        # liquidate.py checks the account number before it closes anything.
        if path == "/v2/account":
            return {"account_number": "PA0TESTACCT1"}
        return [{"symbol": "AAA", "status": 200}, {"symbol": "BBB", "status": 200}]

    monkeypatch.setattr(liq.pt, "api", fake_api)
    return calls


def test_liquidate_without_confirmation_sends_nothing(liq, monkeypatch):
    calls = _api_recorder(monkeypatch, liq)
    monkeypatch.setattr(sys, "argv", ["liquidate.py"])
    assert liq.main() == 0
    assert not any(m == "DELETE" for m, _ in calls), "dry run must not delete"


def test_liquidate_rejects_a_wrong_confirmation(liq, monkeypatch):
    calls = _api_recorder(monkeypatch, liq)
    monkeypatch.setattr(sys, "argv", ["liquidate.py", "--confirm", "yes"])
    assert liq.main() == 0
    assert not any(m == "DELETE" for m, _ in calls)


def test_liquidate_refuses_while_the_market_is_shut(liq, monkeypatch):
    calls = _api_recorder(monkeypatch, liq, is_open=False)
    monkeypatch.setattr(sys, "argv", ["liquidate.py", "--confirm", "LIQUIDATE"])
    assert liq.main() == 1
    assert not any(m == "DELETE" for m, _ in calls)


def test_liquidate_closes_everything_when_confirmed(liq, monkeypatch):
    calls = _api_recorder(monkeypatch, liq)
    monkeypatch.setattr(sys, "argv", ["liquidate.py", "--confirm", "LIQUIDATE"])
    assert liq.main() == 0
    assert ("DELETE", "/v2/positions") in calls


def test_liquidate_is_a_noop_when_already_flat(liq, monkeypatch):
    monkeypatch.setattr(liq.pt, "positions", dict)
    calls = _api_recorder(monkeypatch, liq)
    monkeypatch.setattr(sys, "argv", ["liquidate.py", "--confirm", "LIQUIDATE"])
    assert liq.main() == 0
    assert not any(m == "DELETE" for m, _ in calls)


def test_history_row_is_appended(rd):
    rd.append_history({"date": "2026-09-02", "equity": 101_130.0,
                       "deployed": 5000.0, "position_value": 4990.0,
                       "n_positions": 39, "orders": 42, "sent": 42,
                       "rejected": 0, "turnover": 4600.0})
    rows = list(__import__("csv").DictReader(rd.HISTORY.open()))
    assert len(rows) == 1
    assert rows[0]["position_value"] == "4990.0"
    assert rows[0]["n_positions"] == "39"


# --- the morning email's headline number -----------------------------------
#
# On 2026-09-03 the rebalance email led with "+69.74 (+1.39% of the $5,000
# book)" and no span label at all. That figure was the change since the
# PREVIOUS DAY's rebalance at 14:28 UTC -- nearly 25 hours, most of it
# overnight with the market shut. It read as "what the bot made today".

def test_morning_email_leads_with_the_session_not_the_last_run(monkeypatch):
    captured = {}
    monkeypatch.setattr(notify, "send",
                        lambda subject, html: captured.update(
                            subject=subject, html=html) or True)
    notify.daily_summary(
        equity=101_039.68, deployed=5000.0, orders=[], sent=43, failed=[],
        prev_equity=100_969.94,           # the misleading 25-hour baseline
        session={"pnl": 21.30, "open_equity": 101_018.38,
                 "close_equity": 101_039.68, "n_points": 300},
        since_date="2026-09-02")
    # The session figure wins; the stale one must not appear as the headline.
    assert "+21.30" in captured["subject"]
    assert "today" in captured["subject"]
    assert "69.74" not in captured["subject"]
    assert "market open to now" in captured["html"]


def test_morning_email_labels_the_fallback_span(monkeypatch):
    """No session record: still show the number, but never bare."""
    captured = {}
    monkeypatch.setattr(notify, "send",
                        lambda subject, html: captured.update(
                            subject=subject, html=html) or True)
    notify.daily_summary(
        equity=101_039.68, deployed=5000.0, orders=[], sent=43, failed=[],
        prev_equity=100_969.94, session={}, since_date="2026-09-02")
    assert "+69.74" in captured["subject"]
    assert "since last run" in captured["subject"]
    assert "2026-09-02" in captured["html"]
    assert "not</b> today" in captured["html"]


def test_a_one_point_session_is_not_trusted(monkeypatch):
    """Right at the open there is no span yet -- fall back rather than
    report a P&L computed from a single reading."""
    captured = {}
    monkeypatch.setattr(notify, "send",
                        lambda subject, html: captured.update(
                            subject=subject, html=html) or True)
    notify.daily_summary(
        equity=101_000.0, deployed=5000.0, orders=[], sent=0, failed=[],
        prev_equity=100_900.0,
        session={"pnl": 0.0, "open_equity": 101_000.0,
                 "close_equity": 101_000.0, "n_points": 1},
        since_date="2026-09-02")
    assert "since last run" in captured["subject"]


def test_since_date_is_read_before_record_run_overwrites_it(rd, tmp_path):
    """record_run() stamps TODAY over the state file, so the fallback label
    has to be captured first or it would read 'since today'."""
    rd.record_run("2026-09-03", {"equity": 101_039.68})
    assert rd.last_run_date() == "2026-09-03"
    # Simulating the next session: the value read before record_run is the
    # one the email must quote.
    captured_before = rd.last_run_date()
    rd.record_run("2026-09-04", {"equity": 101_100.0})
    assert captured_before == "2026-09-03"
    assert rd.last_run_date() == "2026-09-04"


# --- history.csv column migration ------------------------------------------
#
# state/history.csv was created on 2026-09-02 with nine columns. PR #6 added
# session_open/session_close/session_pnl to the writer, but the header is only
# written at file creation -- so 09-03's row went in with twelve values under
# the nine-column header and every field after `date` was shifted. The file
# claimed equity was 101,010.36 when that number was the session open.

def test_history_migrates_a_stale_header(rd, tmp_path):
    f = tmp_path / "history.csv"
    f.write_text(
        "date,equity,deployed,position_value,n_positions,orders,sent,"
        "rejected,turnover\n"
        "2026-09-02,100969.94,5000.0,5465.61,47,69,69,0,8222.9\n")
    rd.HISTORY = f
    rd.append_history({"date": "2026-09-03", "session_open": 101_010.36,
                       "session_close": 101_065.52, "session_pnl": 55.16,
                       "equity": 101_062.47, "deployed": 5000.0,
                       "position_value": 5060.75, "n_positions": 43,
                       "orders": 43, "sent": 43, "rejected": 0,
                       "turnover": 3996.98})
    rows = list(__import__("csv").DictReader(f.open()))
    # No ragged rows: DictReader parks overflow under a None key.
    assert all(None not in r for r in rows)
    # The old row keeps its real values and gains empty session cells.
    assert rows[0]["equity"] == "100969.94"
    assert rows[0]["session_pnl"] == ""
    # The new row lands in the right columns.
    assert rows[1]["equity"] == "101062.47"
    assert rows[1]["session_open"] == "101010.36"


def test_history_realigns_a_row_already_written_wide(rd, tmp_path):
    """The damage is already on disk, so migration must repair it, not just
    prevent the next one."""
    f = tmp_path / "history.csv"
    f.write_text(
        "date,equity,deployed,position_value,n_positions,orders,sent,"
        "rejected,turnover\n"
        "2026-09-02,100969.94,5000.0,5465.61,47,69,69,0,8222.9\n"
        "2026-09-03,101010.36,101065.52,55.16,101062.47,5000.0,5060.75,"
        "43,43,43,0,3996.98\n")
    rd.HISTORY = f
    rd.append_history({"date": "2026-09-04", "equity": 1.0, "deployed": 5000.0})
    rows = list(__import__("csv").DictReader(f.open()))
    assert [r["date"] for r in rows] == ["2026-09-02", "2026-09-03", "2026-09-04"]
    # The wide row is re-read positionally against the NEW column order.
    assert rows[1]["equity"] == "101062.47"
    assert rows[1]["session_pnl"] == "55.16"
    assert all(None not in r for r in rows)


def test_history_appends_normally_once_the_header_is_current(rd, tmp_path):
    """The migration path must not rewrite the file on every ordinary run."""
    f = tmp_path / "history.csv"
    rd.HISTORY = f
    rd.append_history({"date": "2026-09-04", "equity": 1.0})
    before = f.read_text()
    rd.append_history({"date": "2026-09-07", "equity": 2.0})
    rows = list(__import__("csv").DictReader(f.open()))
    assert len(rows) == 2
    assert f.read_text().startswith(before)   # pure append, nothing rewritten


# --------------------------------------------------------------------------
# The account pin
#
# Until 2026-09-05 the bot traded a ~$101,000 account with MAX_DEPLOY=5000
# holding the book to $5,000, so credentials pointing at the wrong account
# would still only have risked the capped amount. It now trades a dedicated
# $5,000 account with no cap, which moves that protection entirely onto the
# account number: a wrong-account run would deploy the whole of whatever it
# reached, and every downstream signal -- fills, emails, the equity curve --
# would look perfectly normal. These cover the gate that replaced the cap.
# --------------------------------------------------------------------------

def test_account_pin_accepts_the_expected_account(monkeypatch):
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "PA0TESTACCT1")
    monkeypatch.setattr(pt, "api", lambda *a, **k: {"account_number": "PA0TESTACCT1"})
    assert pt.assert_expected_account() == "PA0TESTACCT1"


def test_account_pin_rejects_a_different_account(monkeypatch):
    """The failure this exists for: keys restored from a backup of the old
    ~$101,000 account, against a config that no longer caps the book."""
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "PA0TESTACCT1")
    monkeypatch.setattr(pt, "api", lambda *a, **k: {"account_number": "PA0OLDACCOUNT"})
    with pytest.raises(RuntimeError) as exc:
        pt.assert_expected_account()
    # Both numbers must appear: the message is what tells a human which half
    # of the pair is wrong, the keys or the pin.
    assert "PA0OLDACCOUNT" in str(exc.value) and "PA0TESTACCT1" in str(exc.value)


def test_account_pin_rejects_an_account_alpaca_will_not_name(monkeypatch):
    """A response missing account_number must fail closed, not pass by
    comparing empty to empty."""
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "PA0TESTACCT1")
    monkeypatch.setattr(pt, "api", lambda *a, **k: {})
    with pytest.raises(RuntimeError):
        pt.assert_expected_account()


def test_account_pin_unset_allows_any_account(monkeypatch):
    """Unpinned is a legitimate local rehearsal; the workflows always pin."""
    monkeypatch.delenv("ALPACA_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(pt, "api", lambda *a, **k: {"account_number": "PAWHATEVER"})
    assert pt.assert_expected_account() == "PAWHATEVER"


def test_blank_account_pin_is_treated_as_unset(monkeypatch):
    """An Actions variable that exists but is empty renders as "", which must
    mean unpinned rather than 'only an account with no number'."""
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "   ")
    monkeypatch.setattr(pt, "api", lambda *a, **k: {"account_number": "PAWHATEVER"})
    assert pt.assert_expected_account() == "PAWHATEVER"


def test_rebalance_refuses_to_plan_for_an_unrecognised_account(monkeypatch, rd):
    """End to end through main(): the gate must sit before the fetch and the
    plan, not next to the orders. By the time a plan exists the run has spent
    twenty minutes and a full universe fetch proving it was aimed at the wrong
    book, and the useful moment to stop has passed."""
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "PA0TESTACCT1")
    # Undo the fixture's stub -- this is the one test that wants the real gate.
    monkeypatch.setattr(rd.pt, "assert_expected_account", _REAL_ACCOUNT_PIN)

    def fake_api(path, method="GET", body=None, base=rd.pt.PAPER_BASE):
        if path == "/v2/clock":
            return {"is_open": True, "next_open": "2026-09-08T09:30:00-04:00"}
        if path == "/v2/account":
            return {"account_number": "PA0OLDACCOUNT"}
        raise AssertionError(f"unexpected broker call: {method} {path}")

    def explode(*a, **k):
        raise AssertionError("a plan must not be built for the wrong account")

    monkeypatch.setattr(rd.pt, "api", fake_api)
    monkeypatch.setattr(rd.pt, "build_plan", explode)
    monkeypatch.setattr(rd.pt, "submit", explode)
    monkeypatch.setattr(rd.pt, "account", explode)

    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--execute"])
    with pytest.raises(RuntimeError, match="account mismatch"):
        rd.main()


def test_summary_refuses_to_report_an_unrecognised_account(monkeypatch, rd):
    """--summary-only places no orders, but it does state a balance as the
    day's result. Reading that off the wrong account is its own kind of wrong
    answer, so it is gated too."""
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "PA0TESTACCT1")
    monkeypatch.setattr(rd.pt, "assert_expected_account", _REAL_ACCOUNT_PIN)
    monkeypatch.setattr(rd.pt, "api", lambda *a, **k: {"account_number": "PA0OLDACCOUNT"})

    sent = []
    monkeypatch.setattr(rd.notify, "close_summary", lambda **kw: sent.append(kw))

    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--summary-only"])
    with pytest.raises(RuntimeError, match="account mismatch"):
        rd.main()
    assert not sent, "no email may quote a balance from the wrong account"


def test_refuses_the_assay_bots_account_even_when_pinned_to_it(monkeypatch):
    """Each bot trades its own dedicated account. Pointing this repo at the
    assay bot's account -- by copying its secrets and variable -- must stop
    the run, or two strategies would fight over one book."""
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "PA318Q9SK8B5")
    monkeypatch.setattr(pt, "api", lambda *a, **k: {"account_number": "PA318Q9SK8B5"})
    with pytest.raises(RuntimeError, match="belongs to another bot"):
        pt.assert_expected_account()


def test_refuses_the_assay_bots_account_when_unpinned(monkeypatch):
    monkeypatch.delenv("ALPACA_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(pt, "api", lambda *a, **k: {"account_number": "PA318Q9SK8B5"})
    with pytest.raises(RuntimeError, match="belongs to another bot"):
        pt.assert_expected_account()


def test_executing_run_requires_an_account_pin(monkeypatch, rd):
    """No default account exists in this repo's workflows, so an executing
    run with the variable unset must stop before anything is fetched."""
    monkeypatch.delenv("ALPACA_ACCOUNT_ID", raising=False)

    def fake_api(path, method="GET", body=None, base=rd.pt.PAPER_BASE):
        if path == "/v2/clock":
            return {"is_open": True, "next_open": "2026-09-08T09:30:00-04:00"}
        raise AssertionError(f"unexpected broker call: {method} {path}")

    def explode(*a, **k):
        raise AssertionError("nothing may be planned without an account pin")

    monkeypatch.setattr(rd.pt, "api", fake_api)
    monkeypatch.setattr(rd.pt, "build_plan", explode)
    monkeypatch.setattr(rd.pt, "submit", explode)
    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--execute"])
    with pytest.raises(RuntimeError, match="ALPACA_ACCOUNT_ID is not set"):
        rd.main()


def _live_panel(n_days=600, n_names=60, seed=3):
    from core.data import Panel
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-06-19", periods=n_days)
    cols = [f"T{i}" for i in range(n_names)]
    drift = rng.normal(0.0006, 0.0006, n_names)
    c = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(drift, 0.015, (n_days, n_names)), 0)),
                     idx, cols)
    v = pd.DataFrame(1e6, idx, cols)
    return Panel(c.shift(1).fillna(c), c * 1.01, c * 0.99, c, v, c,
                 survivorship_free=False, source="yahoo")


def _plan_with(monkeypatch, panel):
    monkeypatch.setattr(pt, "fetch_panel", lambda *a, **k: panel)
    monkeypatch.setattr(pt, "positions", lambda: {})
    monkeypatch.setattr(pt, "position_qtys", lambda: {})
    return pt.build_plan(5000.0)


def test_a_near_empty_last_row_does_not_empty_the_plan(monkeypatch):
    """Reproduces the first live dry run (2026-09-29): Yahoo returned a final
    date with almost no bars, every name looked untradable on it, and the run
    died with 'strategy produced no target positions today'."""
    p = _live_panel()
    extra = p.close.index[-1] + pd.offsets.BDay(1)
    for fld in ["open", "high", "low", "close", "volume", "adj_close"]:
        df = getattr(p, fld)
        row = pd.DataFrame(np.nan, index=[extra], columns=df.columns)
        row.iloc[0, :2] = df.iloc[-1, :2].to_numpy()     # 2 of 60 names printed
        setattr(p, fld, pd.concat([df, row]))
    plan = _plan_with(monkeypatch, p)
    # A partial row must not shrink the book to the few names that printed
    # on it -- that would concentrate the account in 2 stocks.
    assert (plan["weight"] > 0).sum() >= 5
    assert plan["weight"].max() < 0.5


def test_a_near_empty_row_inside_the_history_does_not_empty_the_plan(monkeypatch):
    p = _live_panel()
    d = p.close.index[-30]
    for fld in ["open", "high", "low", "close", "volume", "adj_close"]:
        df = getattr(p, fld).copy()
        df.loc[d, df.columns[2:]] = np.nan
        setattr(p, fld, df)
    plan = _plan_with(monkeypatch, p)
    # A partial row must not shrink the book to the few names that printed
    # on it -- that would concentrate the account in 2 stocks.
    assert (plan["weight"] > 0).sum() >= 5
    assert plan["weight"].max() < 0.5


def _plan_for(monkeypatch, target, held):
    """build_plan against a fixed target and holdings; no network."""
    from core.data import Panel
    syms = sorted(set(target) | set(held))
    idx = pd.bdate_range("2026-01-01", periods=5)
    c = pd.DataFrame(10.0, idx, syms)
    p = Panel(c, c, c, c, c * 1e6, c, survivorship_free=False, source="test")
    monkeypatch.setattr(pt, "fetch_panel", lambda *a, **k: p)
    monkeypatch.setattr(pt, "todays_target", lambda *a, **k: pd.Series(target, dtype=float))
    monkeypatch.setattr(pt, "positions", lambda: dict(held))
    monkeypatch.setattr(pt, "position_qtys", lambda: {k: v / 10.0 for k, v in held.items()})
    return pt.build_plan(5000.0).set_index("symbol")


def test_small_new_positions_are_opened(monkeypatch):
    """Found in the first MA dry run: 305 names at ~$16 each, 0 orders."""
    target = {f"S{i}": 1 / 305 for i in range(305)}
    plan = _plan_for(monkeypatch, target, {})
    assert plan["act"].sum() == 305


def test_small_top_ups_are_still_skipped(monkeypatch):
    plan = _plan_for(monkeypatch, {"A": 0.5, "B": 0.5}, {"A": 2495.0, "B": 2400.0})
    assert not plan.loc["A", "act"]          # $5 off target: leave it
    assert plan.loc["B", "act"]              # $100 off target: trade


def test_a_position_below_the_minimum_can_still_be_closed(monkeypatch):
    plan = _plan_for(monkeypatch, {"A": 1.0}, {"A": 4990.0, "OLD": 12.0})
    assert plan.loc["OLD", "act"]


def test_engine_opens_positions_smaller_than_the_band():
    """Same rule in the backtest: a 0.4% position with a 0.5% band opens."""
    from core.costs import CostModel
    from core.data import Panel
    from core.engine import Backtester
    idx = pd.bdate_range("2024-01-01", periods=60)
    cols = [f"S{i}" for i in range(250)]
    c = pd.DataFrame(20.0, idx, cols)
    p = Panel(c, c, c, c, c * 1e6, c, survivorship_free=True, source="test")
    bt = Backtester(p, CostModel(), initial_capital=5000.0, min_price=3.0,
                    max_price=1e9, min_dollar_volume=5e6, rebalance_band=0.005,
                    spread_model="tiered", allow_short=False, max_gross=1.0)
    w = pd.DataFrame(1 / 250, idx, cols)
    res = bt.run(w)
    assert res.n_positions.iloc[-1] == 250


def test_a_rate_limited_order_is_retried_not_rejected(monkeypatch):
    """~300 first-day orders exceed Alpaca's ~200 requests/minute; a 429 must
    be retried rather than counted as a rejection."""
    import io
    import urllib.error
    calls = []

    def fake_api(path, method="GET", body=None, base=None):
        calls.append(body["symbol"])
        if len(calls) == 1:
            raise urllib.error.HTTPError(path, 429, "Too Many Requests", {}, io.BytesIO(b""))
        return {"id": "abc12345"}

    monkeypatch.setattr(pt, "api", fake_api)
    monkeypatch.setattr(pt.time, "sleep", lambda s: None)
    plan = pd.DataFrame([{"symbol": "AAA", "weight": 0.5, "target_$": 100.0,
                          "current_$": 0.0, "delta_$": 100.0, "price": 10.0,
                          "held_qty": 0.0, "act": True}])
    sent, failed = pt.submit(plan)
    assert (sent, failed) == (1, []) and calls == ["AAA", "AAA"]


def test_orders_are_paced(monkeypatch):
    pauses = []
    monkeypatch.setattr(pt, "api", lambda *a, **k: {"id": "abc12345"})
    monkeypatch.setattr(pt.time, "sleep", lambda s: pauses.append(s))
    plan = pd.DataFrame([{"symbol": f"S{i}", "weight": 0.1, "target_$": 100.0,
                          "current_$": 0.0, "delta_$": 100.0, "price": 10.0,
                          "held_qty": 0.0, "act": True} for i in range(5)])
    assert pt.submit(plan)[0] == 5
    assert pauses == [pt.ORDER_PAUSE_SEC] * 4
