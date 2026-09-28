"""
Tests for the fire-at-the-bell scheduler (market_wait.py).

This module decides whether a trading day happens at all, so its failure modes
are the expensive kind. Two real incidents are pinned here as tests:

  * 2026-09-03: GitHub delivered a scheduled fire at 13:28 UTC, two minutes
    BEFORE the 13:30 open. The old gate saw a shut market and skipped, and the
    day needed a manual dispatch. It must now wait those two minutes instead.
  * The close-of-day email must never be sent mid-session: doing so both
    reports a wrong number and consumes the once-per-day gate, so the real
    close email never goes out.

Nothing here touches the network -- plan_wait is pure, which is why it was
split out of the I/O.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_wait as mw

OPEN = datetime(2026, 9, 3, 13, 30, tzinfo=timezone.utc)
CLOSE = datetime(2026, 9, 3, 20, 0, tzinfo=timezone.utc)


def call(**kw):
    base = dict(now=OPEN, is_open=False, next_open=OPEN, next_close=CLOSE,
                until="open", max_wait_s=25 * 60, grace_s=45.0)
    base.update(kw)
    return mw.plan_wait(**base)


# --- until=open ------------------------------------------------------------

def test_waits_for_an_open_two_minutes_away():
    """The exact 2026-09-03 failure: a fire that landed just before the bell."""
    action, sleep_s, _ = call(now=OPEN - timedelta(minutes=2))
    assert action == "wait"
    # 120s to the bell plus the grace period, not a moment more.
    assert sleep_s == pytest.approx(120 + 45)


def test_runs_immediately_when_the_market_is_already_open():
    action, sleep_s, _ = call(now=OPEN + timedelta(minutes=15), is_open=True)
    assert action == "run"
    assert sleep_s == 0


def test_skips_rather_than_billing_a_long_sleep():
    """A sleeping runner bills like a working one on a private repo."""
    action, _, why = call(now=OPEN - timedelta(hours=3))
    assert action == "skip"
    assert "wait cap" in why


def test_waits_right_up_to_the_cap():
    action, sleep_s, _ = call(now=OPEN - timedelta(minutes=25))
    assert action == "wait"
    assert sleep_s == pytest.approx(25 * 60 + 45)


def test_runs_when_the_open_has_already_passed_but_clock_lags():
    action, _, _ = call(now=OPEN + timedelta(minutes=5), is_open=False)
    assert action == "run"


def test_missing_next_open_fails_open():
    """A clock we cannot read must never be able to stop trading."""
    action, _, _ = call(next_open=None)
    assert action == "run"


# --- until=close -----------------------------------------------------------

def test_close_waits_for_the_bell_with_a_settling_grace():
    action, sleep_s, _ = call(until="close", is_open=True,
                              now=CLOSE - timedelta(minutes=15),
                              grace_s=150.0)
    assert action == "wait"
    # The extra 150s lets Alpaca's final one-minute bar settle before
    # session_pnl() reads it.
    assert sleep_s == pytest.approx(15 * 60 + 150)


def test_close_refuses_to_send_mid_session():
    """Sending early would report a wrong number AND burn the daily gate."""
    action, _, why = call(until="close", is_open=True,
                          now=CLOSE - timedelta(hours=4))
    assert action == "skip"
    assert "too early" in why


def test_close_sends_immediately_once_the_market_is_shut():
    action, sleep_s, _ = call(until="close", is_open=False,
                              now=CLOSE + timedelta(minutes=40))
    assert action == "run"
    assert sleep_s == 0


def test_unknown_mode_is_an_error_not_a_silent_run():
    with pytest.raises(ValueError):
        call(until="lunchtime")


# --- state gate and timestamp parsing -------------------------------------

def test_already_done_reads_only_a_positive_match(tmp_path):
    p = tmp_path / "last_run.json"
    p.write_text(json.dumps({"date": "2026-09-03"}))
    assert mw.already_done(str(p), "2026-09-03") is True
    assert mw.already_done(str(p), "2026-09-04") is False


def test_missing_or_corrupt_state_never_reads_as_done(tmp_path):
    """Mirrors run_daily.already_ran_today: a bad file must not stop the day."""
    assert mw.already_done(str(tmp_path / "nope.json"), "2026-09-03") is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert mw.already_done(str(bad), "2026-09-03") is False


@pytest.mark.parametrize("stamp", [
    "2026-09-03T13:30:00Z",
    "2026-09-03T09:30:00-04:00",
    "2026-09-03T13:30:00.123456789Z",   # nanoseconds: fromisoformat rejects
    "2026-09-03T13:30:00.5-00:00",
])
def test_parse_ts_normalises_to_utc(stamp):
    got = mw.parse_ts(stamp)
    assert got.tzinfo == timezone.utc
    assert (got - OPEN).total_seconds() < 1


# --- the event gate --------------------------------------------------------

@pytest.mark.parametrize("event", ["schedule", "repository_dispatch"])
def test_automated_fires_respect_the_once_per_day_gate(event, tmp_path,
                                                       monkeypatch, capsys):
    """An external clock must not be able to skip the gate and double-trade.

    repository_dispatch is the entry point for a third-party cron service.
    Treating it as "a human pressed the button" would bypass both the
    once-per-day record and the wait, so it is pinned here.
    """
    state = tmp_path / "last_run.json"
    state.write_text(json.dumps({"date": datetime.now(timezone.utc)
                                 .date().isoformat()}))
    monkeypatch.setenv("EVENT", event)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    # No clock stub needed: the state gate returns before any network call,
    # so a request here would be a bug the test would surface as an error.
    monkeypatch.setattr(mw, "fetch_clock",
                        lambda: pytest.fail("clock must not be called"))
    assert mw.main(["--until", "open", "--state", str(state)]) == 0
    assert "run=false" in capsys.readouterr().out


def test_workflow_dispatch_never_waits(monkeypatch, capsys):
    monkeypatch.setenv("EVENT", "workflow_dispatch")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(mw, "fetch_clock",
                        lambda: pytest.fail("clock must not be called"))
    monkeypatch.setattr(mw.time, "sleep",
                        lambda s: pytest.fail("must not sleep"))
    assert mw.main(["--until", "open"]) == 0
    assert "run=true" in capsys.readouterr().out


def test_an_unreachable_clock_lets_the_job_proceed(monkeypatch, capsys):
    """Fail open. A broken waiter must not be able to halt the strategy."""
    monkeypatch.setenv("EVENT", "schedule")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    def boom():
        raise OSError("connection reset")

    monkeypatch.setattr(mw, "fetch_clock", boom)
    assert mw.main(["--until", "open"]) == 0
    assert "run=true" in capsys.readouterr().out


def test_it_actually_sleeps_then_runs(monkeypatch, capsys):
    """End-to-end through main(): clock says pre-open, so it sleeps and runs."""
    monkeypatch.setenv("EVENT", "schedule")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(mw, "fetch_clock", lambda: {
        "timestamp": "2026-09-03T13:28:00Z", "is_open": False,
        "next_open": "2026-09-03T13:30:00Z",
        "next_close": "2026-09-03T20:00:00Z"})
    slept = []
    monkeypatch.setattr(mw.time, "sleep", slept.append)
    assert mw.main(["--until", "open", "--grace-sec", "45"]) == 0
    assert slept == [pytest.approx(165.0)]
    assert "run=true" in capsys.readouterr().out
