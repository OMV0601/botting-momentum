"""
Close every open position. One-shot, manual, deliberately hard to fire.

Use it to start clean (flat) or to stop the strategy completely.

It is NOT part of the daily path and nothing schedules it. The strategy exits
positions itself during the daily run; this is a manual reset, and using it
routinely would be overriding the thing being measured.

SAFETY
------
  - refuses anything but the paper endpoint, via paper_trade.api's own assert
  - requires --confirm LIQUIDATE; there is no default that does the deletion
  - default is a dry run that prints the positions and stops
  - refuses while the market is shut unless --allow-closed, because a market
    order sent into a closed session does not fill at a price anyone has seen
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import paper_trade as pt  # noqa: E402

JOURNAL = ROOT / "journal.md"
CONFIRM = "LIQUIDATE"


def log(msg: str, level: str = "INFO") -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {level:<5} {msg}"
    print(line, flush=True)
    try:
        with JOURNAL.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default="",
                    help=f"must be exactly {CONFIRM!r} to place any order")
    ap.add_argument("--allow-closed", action="store_true",
                    help="permit submission while the market is shut")
    args = ap.parse_args()

    base = pt.resolve_base()
    assert base in (pt.PAPER_BASE, pt.LIVE_BASE), f"unknown endpoint {base!r}"
    pt.assert_key_matches_endpoint(base, os.environ.get("ALPACA_API_KEY_ID", ""))
    live = base == pt.LIVE_BASE
    if live:
        print("*** LIVE — this sells REAL positions for REAL money ***")

    # The right endpoint is not the same question as the right account. This is
    # the one script here that destroys a book rather than adjusting it, so it
    # gets the same pin the daily path has: wrong account, no orders.
    log(f"liquidating Alpaca {'LIVE' if live else 'paper'} account "
        f"{pt.assert_expected_account()}")

    equity, cash, status = pt.account()
    held = pt.positions()
    qtys = pt.position_qtys()
    total = float(sum(held.values()))

    log(f"=== liquidate: {len(held)} positions worth ${total:,.2f} "
        f"(equity ${equity:,.2f}, cash ${cash:,.2f}, status {status}) ===")
    for sym in sorted(held, key=lambda s: -held[s]):
        log(f"  {sym:<6} {qtys.get(sym, 0):>12,.4f} sh  ${held[sym]:>10,.2f}")

    if not held:
        log("already flat; nothing to do")
        return 0

    if args.confirm != CONFIRM:
        log(f"DRY RUN — nothing sent. Re-run with --confirm {CONFIRM} to close "
            f"all {len(held)} positions.", "WARN")
        return 0

    clock = pt.api("/v2/clock")
    if not clock.get("is_open"):
        if not args.allow_closed:
            log(f"market closed (next open {clock.get('next_open','?')}); "
                f"refusing. Market orders need an open session to fill at a "
                f"sane price. Pass --allow-closed to override.", "SKIP")
            return 1
        log("market closed but --allow-closed was passed; proceeding", "WARN")

    # DELETE /v2/positions closes everything in one call, server-side, so there
    # is no partially-iterated state to reason about if this dies mid-way.
    log(f"closing all {len(held)} positions...")
    try:
        result = pt.api("/v2/positions", "DELETE")
    except Exception as exc:
        log(f"liquidation request failed: {exc}", "ERROR")
        return 1

    rows = result if isinstance(result, list) else []
    ok = [r for r in rows if 200 <= int(r.get("status", 0)) < 300]
    bad = [r for r in rows if not (200 <= int(r.get("status", 0)) < 300)]
    for r in bad:
        log(f"  FAILED {r.get('symbol','?')}: {r.get('body')}", "WARN")
    log(f"submitted: {len(ok)} accepted"
        + (f", {len(bad)} rejected" if bad else ""))
    log("orders are market orders and settle during the session; verify the "
        "account is flat before relying on it")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
