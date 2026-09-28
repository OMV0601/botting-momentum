"""
Preflight checks for the trading bot. Places no orders, ever.

Run this first after setting credentials, and any time the scheduled job starts
behaving oddly. It verifies — in order of how early they fail — the things that
actually stop a live run:

  1. core Python dependencies import
  2. ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are set
  3. the universe file paper_trade.py reads at import time exists
  4. the credentials authenticate against the PAPER endpoint
  5. the authenticated account is the one ALPACA_ACCOUNT_ID pins
  6. the account is tradable, unlevered, and not blocked
  7. the market clock is reachable

Exit code 0 means a scheduled run has everything it needs. Non-zero names the
first failing check.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"
UNIVERSE = Path(__file__).resolve().parent / "data" / "alpaca_universe2.csv"

_ok, _fail = [], []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            _ok.append((name, detail or ""))
        except Exception as exc:
            _fail.append((name, str(exc)))
        return fn
    return deco


def _headers():
    k = os.environ.get("ALPACA_API_KEY_ID")
    s = os.environ.get("ALPACA_API_SECRET_KEY")
    if not (k and s):
        raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def _base() -> str:
    """Same rule as paper_trade.resolve_base, deliberately duplicated.

    preflight exists partly to prove numpy and pandas import at all, so it must
    not import paper_trade to find this out -- that would make the dependency
    check depend on the dependencies. Six lines of duplication is the cheaper
    error. tests/test_live_path.py asserts the two agree.
    """
    if os.environ.get("ALPACA_LIVE", "").strip() != "true":
        return PAPER_BASE
    missing = [v for v in ("ALPACA_ACCOUNT_ID", "MAX_DEPLOY")
               if not os.environ.get(v, "").strip()]
    if missing:
        raise RuntimeError(f"ALPACA_LIVE=true but {' and '.join(missing)} not set")
    return LIVE_BASE


def _get(path, base=None):
    base = _base() if base is None else base
    assert base in (PAPER_BASE, LIVE_BASE), f"unknown trading endpoint {base!r}"
    req = urllib.request.Request(base + path, headers=_headers())
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


@check("dependencies")
def _deps():
    import numpy, pandas, scipy  # noqa: F401
    return f"numpy {numpy.__version__}, pandas {pandas.__version__}"


@check("credentials present")
def _creds():
    k = _headers()["APCA-API-KEY-ID"]
    # Never print a secret. Fingerprint only.
    return f"key id {k[:4]}...{k[-4:]} ({len(k)} chars)"


@check("universe file")
def _universe():
    if not UNIVERSE.exists():
        raise FileNotFoundError(
            f"{UNIVERSE} missing — paper_trade.py reads it on every run"
        )
    n = sum(1 for _ in UNIVERSE.open()) - 1
    return f"{n} symbols"


@check("endpoint")
def _endpoint():
    """Which world this run is in. Printed first and unconditionally.

    Without this, a 401 is ambiguous: live keys sent to the paper URL and a
    revoked key both look identical. Naming the endpoint and the key prefix
    separates them at a glance -- AK against paper-api is a configuration
    problem, AK against api.alpaca.markets is a credentials problem.
    """
    base = _base()
    key = os.environ.get("ALPACA_API_KEY_ID", "")
    raw = os.environ.get("ALPACA_LIVE", "")
    world = "LIVE — REAL MONEY" if base == LIVE_BASE else "paper"
    # Deliberately NOT echoing the raw value. These settings may arrive from
    # Secrets rather than Variables, and GitHub masks secret values in logs --
    # printing it would show *** and, worse, mask the word "true" everywhere
    # else in the output. Derived state says the same thing and survives.
    live = ("enabled" if raw == "true"
            else f"NOT enabled ({'empty/unset' if not raw else 'set but not exactly true'})")
    return (f"{world} ({base}), ALPACA_LIVE {live}, "
            f"key prefix {key[:2] or '??'}")


@check("authentication")
def _auth():
    base = _base()
    try:
        a = _get("/v2/account")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            key = os.environ.get("ALPACA_API_KEY_ID", "")
            if base == PAPER_BASE and key.startswith("AK"):
                hint = ("a LIVE key (AK...) was sent to the PAPER endpoint. "
                        "Set the ALPACA_LIVE repository variable to exactly "
                        "'true' — note it must be a Variable, not a Secret.")
            elif base == LIVE_BASE and not key.startswith("AK"):
                hint = ("a non-live key was sent to the LIVE endpoint; the "
                        "secrets still hold paper keys.")
            else:
                hint = ("the endpoint and key prefix agree, so the key itself "
                        "is wrong, revoked, or belongs to another account.")
            raise RuntimeError(f"rejected by Alpaca ({exc.code}): {hint}") from exc
        raise
    return f"account {a['account_number']}, status {a['status']}"


@check("account matches ALPACA_ACCOUNT_ID")
def _account_identity():
    want = os.environ.get("ALPACA_ACCOUNT_ID", "").strip()
    got = str(_get("/v2/account").get("account_number", "")).strip()
    if not want:
        # Not a failure: a local rehearsal against a scratch account is a
        # legitimate reason to leave it unset. The workflows always set it.
        return f"account {got}, unpinned (ALPACA_ACCOUNT_ID not set)"
    if got != want:
        raise RuntimeError(
            f"credentials authenticate {got or '<unknown>'}, but this deployment "
            f"is pinned to {want}. The keys and the pin disagree — fix one."
        )
    return f"account {got}, pinned and matching"


@check("account is tradable and unlevered")
def _account_state():
    a = _get("/v2/account")
    problems = []
    if a.get("trading_blocked"):
        problems.append("trading_blocked")
    if a.get("account_blocked"):
        problems.append("account_blocked")
    if a.get("status") != "ACTIVE":
        problems.append(f"status={a['status']}")
    if problems:
        raise RuntimeError("; ".join(problems))

    # A margin-enabled account is NOT a failure. The strategy's weights sum to
    # exactly 1.0 and are never negative, so it deploys settled cash only
    # whatever the broker would allow. run_daily.py asserts that on the actual
    # plan before any order is sent, which is where leverage could really
    # appear.
    mult = float(a.get("multiplier", 1))
    note = f", multiplier {mult:g}" + (" (margin available, unused)" if mult > 1 else "")
    return (f"equity ${float(a['equity']):,.2f}, cash ${float(a['cash']):,.2f}{note}")


@check("market clock")
def _clock():
    c = _get("/v2/clock")
    state = "OPEN" if c["is_open"] else "closed"
    return f"{state}, next open {c['next_open']}"


def main() -> int:
    print("=" * 70)
    print("ASSAY PREFLIGHT — no orders are placed by this script")
    print("=" * 70)
    for name, detail in _ok:
        print(f"  PASS  {name:<34} {detail}")
    for name, detail in _fail:
        print(f"  FAIL  {name:<34} {detail}")
    print("-" * 70)
    if _fail:
        print(f"{len(_fail)} check(s) failed. A scheduled run would not succeed.")
        return 1
    print(f"All {len(_ok)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
