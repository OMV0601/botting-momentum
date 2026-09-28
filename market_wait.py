#!/usr/bin/env python3
"""Fire at the bell, not whenever GitHub gets around to it.

WHY THIS EXISTS
---------------
GitHub's scheduled workflows are best-effort. This repo measured two separate
failure modes:

  * DELAY -- a fire arrives 36 minutes to 9.5 hours after its cron time
    (Aug 24 - Sep 2).
  * DROPPING -- on 2026-09-03 a 34-fire-a-day schedule delivered exactly two
    fires, and both landed *before* the open (13:28 UTC, two minutes early).
    The clock gate correctly refused both, so the day needed a manual dispatch.

Widening the window fixed neither, and asking GitHub more often made the
dropping worse. The insight is that a delay shifts the *start* of a job, so a
job that starts early and then waits can still act at an exact wall-clock
instant. The 13:28 fire above would have slept 122 seconds and traded at the
open.

So: schedule a handful of fires shortly BEFORE the bell, and have each one
sleep until Alpaca's own clock says it is time. Alpaca is the authority on
market hours -- no DST arithmetic, no holiday calendar, no half-days to
maintain here.

COST
----
This is a private repository, so Actions minutes are metered and a sleeping
job bills like a working one. That is the entire reason `--max-wait-min`
exists and defaults low: a fire that would have to wait longer than the cap
skips instead, and a later cron picks the day up. Keep the crons close to the
bell and the cap small.

SAFETY
------
It places no orders and decides nothing about strategy; `run_daily.py` remains
the authority on whether to trade and re-checks the clock itself after this
returns. Every unexpected failure here answers `run=true` -- a broken waiter
must never be able to silently stop trading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

CLOCK_URL = "https://paper-api.alpaca.markets/v2/clock"

# Alpaca stamps the clock in ISO-8601 with an offset; datetime.fromisoformat
# handles that on 3.11+. Nanosecond precision appears occasionally, which
# fromisoformat rejects, so seconds are truncated to six decimals first.
def parse_ts(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        digits = ""
        while tail and tail[0].isdigit():
            digits, tail = digits + tail[0], tail[1:]
        text = f"{head}.{digits[:6]}{tail}"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def plan_wait(*, now, is_open, next_open, next_close, until,
              max_wait_s, grace_s):
    """Pure decision: ("run"|"wait"|"skip", seconds_to_sleep, reason).

    Split out from the I/O so the schedule's behaviour is testable without a
    network or a real clock -- this is the piece that decides whether a
    trading day happens at all.
    """
    if until == "open":
        if is_open:
            return "run", 0, "market is already open"
        if next_open is None:
            return "run", 0, "clock gave no next_open; letting run_daily decide"
        delta = (next_open - now).total_seconds()
        if delta <= 0:
            return "run", 0, "next_open is in the past"
        if delta > max_wait_s:
            return "skip", 0, (
                f"open is {delta / 60:.0f} min away, more than the "
                f"{max_wait_s / 60:.0f} min wait cap")
        return "wait", delta + grace_s, (
            f"waiting {delta / 60:.1f} min for the open at "
            f"{next_open.isoformat()} (+{grace_s:.0f}s grace)")

    if until == "close":
        # Market shut means today's session is behind us (these crons are
        # weekdays only), so the summary is simply late rather than early.
        if not is_open:
            return "run", 0, "market is already closed"
        if next_close is None:
            return "run", 0, "clock gave no next_close; sending now"
        delta = (next_close - now).total_seconds()
        if delta <= 0:
            return "run", 0, "next_close is in the past"
        if delta > max_wait_s:
            # Sending here would stamp a mid-session number as the day's
            # close AND consume the once-per-day gate, suppressing the real
            # one. Skipping and letting a later cron send is the honest move.
            return "skip", 0, (
                f"close is {delta / 60:.0f} min away, more than the "
                f"{max_wait_s / 60:.0f} min wait cap; too early to call it "
                f"a close-of-day figure")
        return "wait", delta + grace_s, (
            f"waiting {delta / 60:.1f} min for the close at "
            f"{next_close.isoformat()} (+{grace_s:.0f}s grace)")

    raise ValueError(f"unknown --until {until!r}")


def already_done(state_path: str, today: str) -> bool:
    """True only if the state file positively records today.

    Mirrors run_daily.already_ran_today: an unreadable file reads as NOT done,
    so a corrupt file can never silently stop the day.
    """
    try:
        with open(state_path) as fh:
            return json.load(fh).get("date") == today
    except FileNotFoundError:
        return False
    except Exception as exc:  # noqa: BLE001 - any parse error means "not done"
        say(f"{state_path} unreadable ({exc}); treating as not done")
        return False


def say(msg: str) -> None:
    # stderr, because stdout is reserved for key=value lines destined for
    # $GITHUB_OUTPUT, where anything else would be a parse error.
    print(f"[wait] {msg}", file=sys.stderr, flush=True)


def emit(run: bool) -> None:
    line = f"run={'true' if run else 'false'}"
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a") as fh:
            fh.write(line + "\n")
    print(line, flush=True)


def fetch_clock():
    req = urllib.request.Request(CLOCK_URL, headers={
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--until", choices=("open", "close"), required=True)
    ap.add_argument("--max-wait-min", type=float, default=25.0,
                    help="skip rather than sleep longer than this (billed)")
    ap.add_argument("--grace-sec", type=float, default=30.0,
                    help="extra seconds past the bell before acting")
    ap.add_argument("--state", default="",
                    help="state file whose {'date': ...} means today is done")
    args = ap.parse_args(argv)

    # A human pressing the button is never made to wait. Everything else --
    # `schedule`, and `repository_dispatch` from an external clock -- is an
    # automated fire and gets the full treatment: the once-per-day gate and the
    # sleep to the bell. Testing only for "not schedule" here would let an
    # external trigger skip both and double-trade.
    if os.environ.get("EVENT", "schedule") == "workflow_dispatch":
        say("manual dispatch — running now")
        emit(True)
        return 0

    today = datetime.now(timezone.utc).date().isoformat()
    if args.state and already_done(args.state, today):
        say(f"{args.state} already records {today} — nothing to do")
        emit(False)
        return 0

    try:
        clock = fetch_clock()
    except Exception as exc:  # noqa: BLE001 - fail open, never block trading
        say(f"clock unreachable ({exc}); letting the job proceed")
        emit(True)
        return 0

    try:
        now = parse_ts(clock["timestamp"])
        nxt_open = parse_ts(clock["next_open"]) if clock.get("next_open") else None
        nxt_close = parse_ts(clock["next_close"]) if clock.get("next_close") else None
        action, sleep_s, why = plan_wait(
            now=now, is_open=bool(clock.get("is_open")),
            next_open=nxt_open, next_close=nxt_close, until=args.until,
            max_wait_s=args.max_wait_min * 60.0, grace_s=args.grace_sec)
    except Exception as exc:  # noqa: BLE001 - fail open
        say(f"could not read the clock ({exc}); letting the job proceed")
        emit(True)
        return 0

    say(why)
    if action == "skip":
        emit(False)
        return 0
    if action == "wait":
        time.sleep(sleep_s)
        say(f"woke at {datetime.now(timezone.utc).isoformat()}")
    emit(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
