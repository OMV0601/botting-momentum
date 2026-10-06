"""
Scheduled entrypoint. Runs once per trading day, at the open.

WHY DAILY. The strategy's backtest assumes one rebalance per trading day (see
strategy.py). Skipping days makes the live book drift away from what was
tested.

This wraps paper_trade.py rather than reimplementing it. All strategy and order
logic still lives there; what is added here is only what a unattended loop
needs and upstream has no opinion about:

  - market-open gate, so holidays and weekends are a clean no-op
  - once-per-day gate, so a retry or a double-fire cannot double-trade
  - a kill switch that needs no code change
  - pre-trade assertions: paper endpoint, no shorts, no leverage
  - an append-only journal.md record of every decision

Default is a DRY RUN. Orders are placed only with --execute.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import notify  # noqa: E402
import paper_trade as pt  # noqa: E402

STATE = ROOT / "state" / "last_run.json"
SUMMARY_STATE = ROOT / "state" / "last_summary.json"
ALERT_STATE = ROOT / "state" / "last_alert.json"
HISTORY = ROOT / "state" / "history.csv"
JOURNAL = ROOT / "journal.md"
HALT = ROOT / "HALT"

# Trading days without a completed run before the dead man's switch fires.
# The sleeve structure assumes one run per trading day, so two consecutive
# misses is already a degraded risk profile -- not a situation to discover a
# week later from a screenshot.
STALE_AFTER_DAYS = int(os.environ.get("STALE_AFTER_DAYS", "2"))

# Weight above which a single name is called out. The strategy does not cap
# weights, and this does not either by default -- it records the fact. Set
# ENFORCE_MAX_WEIGHT=true to make it abort instead.
MAX_WEIGHT = float(os.environ.get("MAX_WEIGHT", "0.10"))


def log(msg: str, level: str = "INFO") -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {level:<5} {msg}"
    print(line, flush=True)
    with JOURNAL.open("a") as fh:
        fh.write(line + "\n")


def already_ran_today(today: str) -> bool:
    if not STATE.exists():
        return False
    try:
        return json.loads(STATE.read_text()).get("date") == today
    except (json.JSONDecodeError, OSError):
        # A corrupt state file must not be read as "already ran" -- that would
        # silently skip trading forever.
        log(f"{STATE} unreadable, treating as no run today", "WARN")
        return False


def notify_safe(fn, *args, **kwargs):
    """Run a notification and swallow anything it throws.

    notify.send already guards its own network call, but the summary builders
    do formatting and arithmetic that could raise on unexpected input. Trading
    has already happened by the time these are called, so an exception here
    would report failure for a run that actually succeeded -- and turn the
    workflow red for a cosmetic reason.

    Returns whatever the notifier returned (the senders return True on
    success), or None if it raised. Callers use that to decide whether a
    notification actually went out -- the close-of-day gate must not record a
    send that never happened.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log(f"notification failed ({type(exc).__name__}: {exc}) — "
            f"trading was unaffected", "WARN")
    return None


def previous_equity() -> float | None:
    """Equity recorded by the last executed run, for the day-over-day figure.

    Read before record_run overwrites it. Returns None on a first run or an
    unreadable file -- the email then simply omits the change line.
    """
    if not STATE.exists():
        return None
    try:
        v = json.loads(STATE.read_text()).get("equity")
        return float(v) if v else None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def record_run(today: str, payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"date": today, **payload}, indent=2) + "\n")


def last_run_date() -> str | None:
    """Date of the last run that actually completed, or None."""
    if not STATE.exists():
        return None
    try:
        return json.loads(STATE.read_text()).get("date")
    except (json.JSONDecodeError, OSError):
        return None


def trading_days_between(a: date, b: date) -> int:
    """Weekdays from a to b, exclusive of a. Holidays are not modelled.

    This only sizes an alert threshold, so overcounting by the odd market
    holiday is harmless -- it makes the switch marginally more eager, which is
    the safe direction for something whose job is to break a silence.
    """
    n, cur = 0, a
    while cur < b:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            n += 1
    return n


def check_staleness(today: str, reason: str) -> None:
    """Break the silence when the bot has stopped trading.

    The failure this exists for: on 2026-08-27 and 08-28 the scheduled runs
    fired hours after the close, hit the market-clock gate, and returned 0
    without sending anything. Four trading days passed with the bot silent, and
    silence looked identical to success. A missed day is exactly the failure the
    sleeve structure cannot absorb, so it has to be the loud one.

    Fires at most once per day: with the schedule firing every 30 minutes, a
    per-invocation alert would be an inbox denial-of-service.
    """
    last = last_run_date()
    if last is None:
        return
    try:
        gap = trading_days_between(date.fromisoformat(last),
                                   date.fromisoformat(today))
    except ValueError:
        return
    if gap < STALE_AFTER_DAYS:
        return

    try:
        if ALERT_STATE.exists() and json.loads(ALERT_STATE.read_text()).get("date") == today:
            return
    except (json.JSONDecodeError, OSError):
        pass

    log(f"STALE: no completed run since {last} ({gap} trading days). "
        f"Latest attempt: {reason}", "WARN")
    notify_safe(notify.failure_alert, "dead man's switch",
                f"No completed rebalance since {last} — {gap} trading days ago.\n\n"
                f"Most recent attempt ended: {reason}\n\n"
                f"The strategy has no stop-losses; positions only leave the book "
                f"during the daily run. While this is stale, the live book is "
                f"drifting away from what the strategy wants.")
    try:
        ALERT_STATE.parent.mkdir(parents=True, exist_ok=True)
        ALERT_STATE.write_text(json.dumps({"date": today, "last_run": last,
                                           "gap_days": gap}) + "\n")
    except OSError as exc:
        log(f"could not record alert state: {exc}", "WARN")


def append_history(row: dict) -> None:
    """One row per completed run, so the account has a readable history.

    journal.md is append-only prose and cannot be plotted or scanned. The
    position-value column is the point: it makes over-deployment visible on the
    day it happens rather than via a screenshot a week later.
    """
    cols = ["date", "session_open", "session_close", "session_pnl",
            "equity", "deployed", "position_value", "n_positions",
            "orders", "sent", "rejected", "turnover"]
    try:
        HISTORY.parent.mkdir(parents=True, exist_ok=True)

        # Migrate a stale header before appending. The header is written once,
        # at file creation, so adding a column to `cols` later silently starts
        # writing wide rows under the narrow old header -- which is exactly
        # what happened on 2026-09-03: history.csv had been created the day
        # before with nine columns, then the three session_* columns were
        # added, and the day's row went in with twelve values against a
        # nine-column header. Every column after `date` was then misaligned,
        # so the file said equity was 101,010.36 when that was the session
        # open. A corrupt long-term record is worse than no record, and this
        # is the one artefact meant to still be trustworthy in a year.
        #
        # Rewriting is safe and cheap: one row per trading day. Old rows keep
        # their values and get empty cells for columns that did not exist.
        existing = []
        if HISTORY.exists():
            with HISTORY.open(newline="") as fh:
                r = csv.reader(fh)
                header = next(r, None)
                if header and header != cols:
                    log(f"history header {header} != {cols}; migrating", "WARN")
                    # restkey catches values from rows already written wide.
                    for raw in r:
                        existing.append(dict(zip(cols, raw))
                                        if len(raw) >= len(cols)
                                        else dict(zip(header, raw)))
                elif header is None:
                    pass  # empty file; fall through and write a fresh header
                else:
                    existing = None  # header already correct: plain append

        if existing is None:
            with HISTORY.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=cols).writerow(
                    {k: row.get(k, "") for k in cols})
        else:
            existing.append({k: row.get(k, "") for k in cols})
            with HISTORY.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for prev in existing:
                    w.writerow({k: prev.get(k, "") for k in cols})
    except OSError as exc:
        log(f"could not append history: {exc}", "WARN")


def summary_sent_today(today: str) -> bool:
    """Has the close-of-day email already gone out today?

    The close-summary workflow carries two crons -- 20:10 UTC for EDT and 21:10
    for EST -- because GitHub cron has no timezone handling. Both fire every
    weekday regardless of which one is correct, exactly like the trading
    schedule does. The difference is that the trading path has a once-per-day
    gate to absorb the extra fire and this had none, so 2026-09-02 produced two
    close-of-day emails ~40 minutes apart.

    Gating on a recorded send rather than on the clock keeps the second fire
    useful: if the first one failed to send, nothing is recorded and the second
    becomes the retry. That is the same reason the coverage refusal does not
    consume the trading day.
    """
    if not SUMMARY_STATE.exists():
        return False
    try:
        return json.loads(SUMMARY_STATE.read_text()).get("date") == today
    except (json.JSONDecodeError, OSError):
        # Match already_ran_today: a corrupt file must read as "not sent", so a
        # bad file can never silently suppress the summary forever.
        log(f"{SUMMARY_STATE} unreadable, treating as not sent today", "WARN")
        return False


def record_summary(today: str) -> None:
    try:
        SUMMARY_STATE.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_STATE.write_text(json.dumps({"date": today}) + "\n")
    except OSError as exc:
        log(f"could not record summary state: {exc}", "WARN")


def close_of_day_summary(today: str) -> int:
    """Email the day's P&L after the close. Places no orders, ever.

    Deliberately a separate entrypoint rather than a flag threaded through
    main(): the close-of-day path must not be able to reach build_plan() or
    submit() even by accident, and the clearest way to guarantee that is for it
    never to call them.
    """
    if summary_sent_today(today):
        log("close-of-day summary already sent today; skipping the duplicate "
            "timezone fire", "SKIP")
        return 0

    # Places no orders, but it does state a balance as the day's result. A
    # figure read off the wrong account is worse than no email at all.
    acct_no = pt.assert_expected_account()
    log(f"summarising Alpaca {'LIVE' if pt.live_enabled() else 'paper'} account {acct_no}")

    equity, cash, status = pt.account()
    held = pt.positions()
    pos_value = float(sum(held.values()))
    st = {}
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text())
        except (json.JSONDecodeError, OSError):
            st = {}

    traded_today = st.get("date") == datetime.now(timezone.utc).date().isoformat()

    # The day's real result, read back from Alpaca's own minute-by-minute
    # equity curve rather than measured from whenever this job happened to run.
    session = pt.session_pnl()
    if session:
        log(f"session P&L: {session['pnl']:+,.2f} "
            f"(${session['open_equity']:,.2f} at the open -> "
            f"${session['close_equity']:,.2f} at the close, "
            f"{session['n_points']} minute marks)")
    else:
        log("session P&L unavailable; the email will fall back to the "
            "since-last-run figure and say so", "WARN")

    log(f"close-of-day: equity=${equity:,.2f} positions=${pos_value:,.2f} "
        f"across {len(held)} names (traded today: {traded_today})")

    ok = notify_safe(notify.close_summary, equity=equity, cash=cash,
                position_value=pos_value, n_positions=len(held),
                deployed=float(st.get("deployed") or 0.0),
                open_equity=float(st.get("equity") or 0.0) or None,
                traded_today=traded_today,
                # The baseline equity comes from the last COMPLETED run, which
                # is only "this morning" if one ran today. Pass the date so the
                # email can name the real period instead of implying a session.
                since_date=st.get("date"),
                session=session,
                orders_sent=int(st.get("sent") or 0),
                rejected=int(st.get("rejected") or 0))

    # Append the session figures to the daily record. Recorded here rather than
    # at rebalance time because only after the close is the day's result known,
    # and a series measured over a consistent span is the point.
    if session:
        append_history({"date": today, "equity": round(equity, 2),
                        "deployed": round(float(st.get("deployed") or 0.0), 2),
                        "position_value": round(pos_value, 2),
                        "n_positions": len(held),
                        "session_open": round(session["open_equity"], 2),
                        "session_close": round(session["close_equity"], 2),
                        "session_pnl": round(session["pnl"], 2),
                        "orders": st.get("orders", ""), "sent": st.get("sent", ""),
                        "rejected": st.get("rejected", ""),
                        "turnover": st.get("turnover", "")})

    # Only claim the day once an email actually went out, so a failed send
    # leaves the second timezone fire free to retry.
    if ok:
        record_summary(today)
    else:
        log("close-of-day email did not send; leaving the day open so the "
            "later fire retries", "WARN")
    return 0


def market_is_open() -> tuple[bool, str]:
    c = pt.api("/v2/clock")
    return bool(c["is_open"]), c.get("next_open", "?")


def assert_safe_to_trade(plan) -> None:
    """Fail loudly rather than send something the rules forbid."""
    # Which world this run is in, stated out loud. resolve_base() returns the
    # paper endpoint unless ALPACA_LIVE=true was set deliberately, and raises
    # if live was asked for without an account pin and a cap.
    base = pt.resolve_base()
    assert base in (pt.PAPER_BASE, pt.LIVE_BASE), f"unknown endpoint {base!r}"
    pt.assert_key_matches_endpoint(base, os.environ.get("ALPACA_API_KEY_ID", ""))
    if base == pt.LIVE_BASE:
        # Real money. Say so in the journal, every single run, so that no
        # reading of this log is ever ambiguous about what was at stake.
        log("*** LIVE TRADING — REAL MONEY — "
            f"account {os.environ.get('ALPACA_ACCOUNT_ID')} "
            f"capped at ${float(os.environ['MAX_DEPLOY']):,.2f} ***", "LIVE")
    else:
        log("paper endpoint (set ALPACA_LIVE=true for real money)")

    shorts = plan[plan["weight"] < 0]
    if not shorts.empty:
        raise RuntimeError(
            f"strategy produced short weights for {list(shorts['symbol'])}; "
            "the strategy is long-only and shorting requires margin"
        )

    # Leverage is measured on what the plan DEPLOYS, not on what the broker
    # would permit. An Alpaca margin account reports multiplier=4 even when
    # every dollar traded is settled cash, so failing on the multiplier
    # blocks a strategy that is unlevered by construction.
    #
    # strategy.py normalizes each sleeve to sum to 1 and divides by HOLD_DAYS,
    # so the 8 sleeves sum to exactly 1.0 -- verified empirically: gross
    # exposure min/max/mean all 1.0000 over a 120-day sample, no negative
    # weights. This asserts that property holds on the live plan too, which
    # is the thing that actually matters.
    gross = float(plan["weight"].sum())
    if gross > 1.01:
        raise RuntimeError(
            f"plan gross exposure is {gross:.4f} of equity; the strategy "
            "is unlevered and must not exceed 1.0"
        )

    a = pt.api("/v2/account")
    mult = float(a.get("multiplier", 1))
    if mult > 1:
        log(f"account permits {mult:g}x margin; plan deploys {gross:.2%} of the "
            f"capital it was handed, so no leverage is used", "INFO")

    heavy = plan[plan["weight"] > MAX_WEIGHT]
    for _, r in heavy.iterrows():
        log(f"concentration: {r['symbol']} at {r['weight']:.2%} exceeds {MAX_WEIGHT:.0%}", "WARN")
    if not heavy.empty and os.environ.get("ENFORCE_MAX_WEIGHT", "").lower() == "true":
        raise RuntimeError(
            f"{len(heavy)} name(s) above MAX_WEIGHT={MAX_WEIGHT:.0%} and "
            "ENFORCE_MAX_WEIGHT is set"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="place orders (default is a dry run)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the once-per-day gate")
    ap.add_argument("--summary-only", action="store_true",
                    help="email the close-of-day P&L and exit; places no orders")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).date().isoformat()

    if args.summary_only:
        log(f"=== {pt.NAME} close-of-day summary {today} ===")
        return close_of_day_summary(today)

    log(f"=== {pt.NAME} daily run {today} (execute={args.execute}) ===")

    if HALT.exists():
        log(f"HALT file present at {HALT} — refusing to trade. Delete it to resume.", "HALT")
        check_staleness(today, "HALT file present")
        return 0

    if os.environ.get("TRADING_ENABLED", "true").lower() == "false":
        log("TRADING_ENABLED=false — refusing to trade.", "HALT")
        check_staleness(today, "TRADING_ENABLED=false")
        return 0

    if args.execute and not args.force and already_ran_today(today):
        log("already executed today; skipping to avoid double-trading", "SKIP")
        return 0

    try:
        is_open, next_open = market_is_open()
    except Exception as exc:
        log(f"could not reach the Alpaca clock: {exc}", "ERROR")
        return 1

    # The clock gates ORDERS, not the dry run. A dry run sends nothing, so
    # letting it proceed with the market closed is what makes it possible to
    # smoke-test the whole pipeline -- auth, data fetch, signal, plan -- on a
    # weekend instead of discovering a break at 09:35 on Monday. Prices will be
    # the last close, which is fine for a rehearsal and useless for a fill.
    if not is_open:
        if args.execute:
            log(f"market closed; next open {next_open}", "SKIP")
            # This exact path ran silently on 2026-08-27 and 08-28, when GitHub
            # fired the schedule hours late. It is where the four missed days
            # went unnoticed, so it is the one that must speak up.
            check_staleness(today, f"market closed (next open {next_open})")
            return 0
        log(f"market closed; continuing anyway because this is a dry run "
            f"(next open {next_open})", "WARN")

    # Before anything is fetched, planned or sent: is this the account this
    # deployment is supposed to be trading? MAX_DEPLOY used to be what kept a
    # run against the wrong (much larger) account from deploying it in full;
    # with the cap cleared so that account equity IS the strategy's equity,
    # the account number is the thing that has to be checked instead.
    if args.execute and not os.environ.get("ALPACA_ACCOUNT_ID", "").strip():
        raise RuntimeError(
            "ALPACA_ACCOUNT_ID is not set. An executing run must be pinned to "
            "this bot's own dedicated account; set the repository variable.")
    acct_no = pt.assert_expected_account()
    log(f"trading Alpaca {'LIVE' if pt.live_enabled() else 'paper'} account {acct_no}")

    prev_equity = previous_equity()
    # Captured here, next to prev_equity and for the same reason: record_run()
    # below overwrites state/last_run.json with TODAY, so reading the date
    # after that point would label the fallback figure "since today" -- which
    # is both wrong and exactly the kind of unlabelled span being fixed.
    since_date = last_run_date()
    equity, cash, status = pt.account()
    log(f"account status={status} equity=${equity:,.2f} cash=${cash:,.2f}")

    # MAX_DEPLOY makes the strategy size as if the account were smaller than it
    # is. Weights are fractions of whatever capital they are handed, so passing
    # a capped figure to build_plan scales every position down proportionally
    # and leaves name selection and relative sizing untouched.
    #
    # The cap is a ceiling, never a floor: if the account is worth less than
    # MAX_DEPLOY, real equity is used and nothing is inflated.
    deploy = equity
    raw_cap = os.environ.get("MAX_DEPLOY", "").strip()
    if raw_cap:
        try:
            cap = float(raw_cap)
        except ValueError:
            log(f"MAX_DEPLOY={raw_cap!r} is not a number; using full equity", "WARN")
        else:
            if cap <= 0:
                log(f"MAX_DEPLOY={cap} is not positive; using full equity", "WARN")
            elif cap < equity:
                deploy = cap
                log(f"MAX_DEPLOY: sizing as if the account held ${deploy:,.2f}, "
                    f"not its real ${equity:,.2f}")
            else:
                log(f"MAX_DEPLOY=${cap:,.2f} is above real equity "
                    f"${equity:,.2f}; using real equity")

    plan = pt.build_plan(deploy)
    assert_safe_to_trade(plan)

    acting = plan[plan["act"]]
    turnover = acting["delta_$"].abs().sum()
    log(f"plan: {len(acting)} orders, ${turnover:,.2f} turnover "
        f"({turnover / deploy:.2%} of deployed capital), "
        f"{int((plan['weight'] > 0).sum())} target names")
    for _, r in acting.iterrows():
        side = "BUY " if r["delta_$"] > 0 else "SELL"
        log(f"  {side} {r['symbol']:<6} weight={r['weight']:.2%} delta=${r['delta_$']:+,.2f}")

    order_rows = [{"symbol": str(r["symbol"]), "weight": float(r["weight"]),
                   "delta": float(r["delta_$"])} for _, r in acting.iterrows()]

    # The day so far, open to now, from Alpaca's own minute curve. Read here
    # rather than differencing against the last run: that baseline spans
    # whenever the bot last happened to fire, which on 2026-09-03 meant a
    # 25-hour window reported as if it were the day's trading. Returns {} if
    # the series is too short to mean anything -- notify falls back and says so.
    session = notify_safe(pt.session_pnl) or {}

    if not args.execute:
        log("DRY RUN — nothing sent")
        notify_safe(notify.daily_summary, equity=equity, deployed=deploy,
                    orders=order_rows, sent=0, failed=[],
                    prev_equity=prev_equity, dry_run=True,
                    session=session, since_date=since_date)
        return 0

    sent, failed = pt.submit(plan)
    for sym, reason in failed:
        log(f"order rejected: {sym} — {reason}", "WARN")

    # Record the run before emailing: the once-per-day gate protects the
    # account, so it must not depend on a notification succeeding.
    record_run(today, {"orders": len(acting), "turnover": round(float(turnover), 2),
                       "equity": round(float(equity), 2),
                       "deployed": round(float(deploy), 2),
                       "sent": sent, "rejected": len(failed)})
    log(f"run recorded in {STATE.relative_to(ROOT)}")

    # The history row is written by the close-of-day summary, not here. A row
    # per day needs the day's result, and that is only known after the close --
    # writing one at rebalance time would either duplicate the day or record a
    # P&L measured over whatever span the schedule happened to produce.

    notify_safe(notify.daily_summary, equity=equity, deployed=deploy,
                orders=order_rows, sent=sent, failed=failed,
                prev_equity=prev_equity, session=session,
                since_date=since_date)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        tb = traceback.format_exc()
        log("unhandled exception:\n" + tb, "ERROR")
        # A silent failure is the dangerous one: the sleeve structure assumes a
        # run every trading day, so a crash needs to reach a human.
        notify_safe(notify.failure_alert, "run_daily", tb)
        # A crash is also a missed day. The per-crash alert says what broke;
        # the switch says how long it has been broken, which is the number that
        # decides whether this is noise or a standing outage.
        try:
            check_staleness(datetime.now(timezone.utc).date().isoformat(),
                            tb.strip().splitlines()[-1][:200])
        except Exception:
            pass
        sys.exit(1)
