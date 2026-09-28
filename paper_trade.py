"""
Paper-trading harness: turns strategy.todays_target() into Alpaca orders.

Adapted from the assay bot's live path (OMV0601/botting-it-up). Everything
here is strategy-agnostic -- it asks strategy.py for target weights and trades
the difference against current holdings. The strategy itself lives only in
strategy.py.

SAFETY
------
Placing an order is an irreversible action, so this script does NOT place one
unless you pass --execute. The default is a dry run that prints exactly what
would be sent and stops. Check the plan, then run it again with --execute if
you want it filled.

The endpoint defaults to paper and only becomes live when ALPACA_LIVE=true is
set deliberately. Live additionally requires ALPACA_ACCOUNT_ID (which account)
and MAX_DEPLOY (how much), and the API key prefix must match the endpoint --
Alpaca issues paper keys as PK... and live keys as AK..., so a key/endpoint
mismatch is the signature of a half-finished switch and aborts the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import AlpacaSource, Panel
from strategy import DESCRIPTION, NAME, todays_target

PAPER_BASE = "https://paper-api.alpaca.markets"

# Accounts that belong to OTHER bots. Each strategy must trade its own
# dedicated account, so that account's balance is that strategy's result.
# PA318Q9SK8B5 is the assay bot's account (OMV0601/botting-it-up).
OTHER_BOTS_ACCOUNTS = {"PA318Q9SK8B5"}
LIVE_BASE = "https://api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


def live_enabled() -> bool:
    """True only for a deliberate, explicit opt-in.

    Anything other than the exact string "true" means paper. A missing
    variable, an empty one, "1", "yes", "True " with a stray space -- all
    paper. Real money should never be reachable by a typo.

    Deliberately NOT stripped. Whitespace tolerance can only ever turn live ON
    by accident, never off, so the failure it introduces is the expensive one.
    A trailing newline out of a secrets store leaves this on paper, which is
    the direction to fail in.
    """
    return os.environ.get("ALPACA_LIVE", "") == "true"


def resolve_base() -> str:
    """The trading endpoint for this run: paper unless live is opted into.

    Live carries two extra requirements, both of which exist because the
    failure they prevent is expensive and silent:

      ALPACA_ACCOUNT_ID  which account. assert_expected_account() already
                         enforces it, but requiring it HERE means a live run
                         cannot even start unpinned.
      MAX_DEPLOY         how much. Without a cap the strategy deploys 100% of
                         whatever balance it finds, so credentials pointing at
                         a larger account than intended would trade all of it
                         and nothing downstream would look wrong.
    """
    if not live_enabled():
        return PAPER_BASE
    missing = [v for v in ("ALPACA_ACCOUNT_ID", "MAX_DEPLOY")
               if not os.environ.get(v, "").strip()]
    if missing:
        raise RuntimeError(
            f"ALPACA_LIVE=true but {' and '.join(missing)} not set. Live "
            "trading requires an explicit account pin and an explicit cap; "
            "refusing to trade real money without both.")
    return LIVE_BASE


def headers():
    k = os.environ.get("ALPACA_API_KEY_ID")
    s = os.environ.get("ALPACA_API_SECRET_KEY")
    if not (k and s):
        sys.exit("set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s,
            "Content-Type": "application/json"}


def assert_key_matches_endpoint(base: str, key: str) -> None:
    """Alpaca issues paper keys as PK... and live keys as AK....

    A mismatch means the endpoint and the credentials disagree about which
    world this is -- the signature of a half-finished switch. Paper keys on the
    live endpoint simply fail to authenticate, which is harmless. LIVE KEYS ON
    THE PAPER ENDPOINT IS THE DANGEROUS DIRECTION and is why this checks both.
    """
    if base == LIVE_BASE and not key.startswith("AK"):
        raise RuntimeError(
            "live endpoint with a non-live API key (expected AK...); the "
            "secrets have not been switched over")
    if base == PAPER_BASE and key.startswith("AK"):
        raise RuntimeError(
            "paper endpoint with a LIVE API key (AK...); refusing to run "
            "with real credentials against a paper URL")


def api(path, method="GET", body=None, base=None):
    base = resolve_base() if base is None else base
    assert base in (PAPER_BASE, LIVE_BASE), f"unknown trading endpoint {base!r}"
    if base == LIVE_BASE and not live_enabled():
        raise RuntimeError("live endpoint without ALPACA_LIVE=true")
    assert_key_matches_endpoint(base, os.environ.get("ALPACA_API_KEY_ID", ""))
    req = urllib.request.Request(base + path, headers=headers(), method=method,
                                 data=json.dumps(body).encode() if body else None)
    with urllib.request.urlopen(req, timeout=45) as r:
        txt = r.read().decode()
    return json.loads(txt) if txt else {}


def account():
    a = api("/v2/account")
    return float(a["equity"]), float(a["cash"]), a["status"]


def assert_expected_account() -> str:
    """Refuse to act on any account but the one this deployment is pinned to.

    The forward test lives or dies on the account number being the one whose
    balance is the strategy's own equity. Before 2026-09-05 the bot traded a
    ~$101,000 account with MAX_DEPLOY holding the book down to $5,000, so the
    account's percentage move meant nothing on its own; it now trades a
    dedicated account funded to exactly $5,000 with no cap, and the account
    balance IS the result.

    That swap moves a safety property out of MAX_DEPLOY and into this check.
    With the cap cleared, a run against the wrong account -- a secret restored
    from a backup, a half-finished key rotation -- would deploy every dollar of
    it instead of $5,000. There is no in-band way to notice that: the orders
    would fill, the emails would look normal, and the equity curve would be
    someone else's. So the account number is checked, and a mismatch stops the
    run before a plan is even built.

    ALPACA_ACCOUNT_ID unset means unpinned, which is only right for a local
    rehearsal against a throwaway account -- the workflows always set it.
    """
    want = os.environ.get("ALPACA_ACCOUNT_ID", "").strip()
    got = str(api("/v2/account").get("account_number", "")).strip()
    if got in OTHER_BOTS_ACCOUNTS:
        raise RuntimeError(
            f"account {got} belongs to another bot. Each strategy trades its own "
            "dedicated Alpaca account; give this repo credentials for its own.")
    if not want:
        return got
    if got != want:
        raise RuntimeError(
            f"account mismatch: credentials authenticate account {got or '<unknown>'}, "
            f"but this deployment is pinned to {want}. Refusing to trade. "
            "Either the ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY secrets point at "
            "the wrong account, or ALPACA_ACCOUNT_ID needs updating to match the "
            "account you intend to trade."
        )
    return got


def positions():
    return {p["symbol"]: float(p["market_value"]) for p in api("/v2/positions")}


def session_pnl() -> dict:
    """P&L from the market open to the close, as Alpaca recorded it minute by
    minute -- not from whenever this job happened to run.

    The close-of-day email used to lead with `equity now - equity when the bot
    last rebalanced`. That baseline moves: the rebalance ran 14:24 UTC on
    2026-09-02 but could run hours later on a day GitHub's scheduler lags, and
    the reading itself lands whenever the summary job fires, which on 09-02 was
    2.5 hours after the close and therefore included after-hours drift. Two
    numbers measured over different spans are not comparable day to day, which
    makes the series useless for judging the strategy over months.

    Alpaca keeps the equity curve, so the honest figure can be read back rather
    than approximated. extended_hours=false restricts the series to the regular
    session, so the first and last points are the open and the close.

    Returns {} when the series is unavailable or too short to be meaningful --
    callers fall back rather than report a number they cannot stand behind.
    """
    try:
        hist = api("/v2/account/portfolio/history"
                   "?period=1D&timeframe=1Min&extended_hours=false")
    except Exception as exc:
        print(f"[session] portfolio history unavailable: {exc}", flush=True)
        return {}

    stamps = hist.get("timestamp") or []
    equity = hist.get("equity") or []
    # Alpaca pads the series with nulls outside trading; pair them up and keep
    # only the minutes that actually carry an equity mark.
    points = [(t, float(e)) for t, e in zip(stamps, equity) if e is not None]
    if len(points) < 2:
        print(f"[session] only {len(points)} equity point(s); "
              f"not enough for an open-to-close figure", flush=True)
        return {}

    open_ts, open_eq = points[0]
    close_ts, close_eq = points[-1]
    return {
        "open_equity": open_eq,
        "close_equity": close_eq,
        "pnl": close_eq - open_eq,
        "open_ts": int(open_ts),
        "close_ts": int(close_ts),
        "n_points": len(points),
    }


def position_qtys():
    """symbol -> shares actually held.

    Sell size has to come from the broker's share count, not from a dollar
    amount divided by a price. See the note in submit().
    """
    return {p["symbol"]: abs(float(p["qty"])) for p in api("/v2/positions")}


def fetch_panel(lookback_days=800):
    """Enough history for the strategy's longest lookback plus warm-up.

    DATA_SOURCE picks the feed. It defaults to yahoo, which is a deliberate
    divergence from upstream:

    AlpacaSource requests feed=sip (core/data.py), and SIP needs a paid
    Alpaca data subscription. On a free account every request comes back
    HTTP 403 -- verified, all 3,998 symbols -- and the fetch dies with
    "Alpaca returned no bars".

    Alpaca's free IEX feed is not a substitute here. IEX is a single venue at
    a few percent of consolidated volume, and strategy.py filters on
    MIN_DOLLAR_VOLUME = 5e6 computed from consolidated volume; against
    IEX-only volume that filter rejects nearly everything.

    Yahoo reports consolidated volume, so the eligibility filter behaves as
    the strategy intends. The cost is that Yahoo is survivor-only, which
    biases a BACKTEST but does not affect today's signal over names that are
    currently listed and tradable. Set DATA_SOURCE=alpaca to switch back if a
    SIP subscription is ever added.
    """
    start = (date.today() - timedelta(days=int(lookback_days * 1.5))).isoformat()
    end = date.today().isoformat()
    syms = pd.read_csv("data/alpaca_universe2.csv")["Symbol"].astype(str).tolist()
    which = os.environ.get("DATA_SOURCE", "yahoo").lower()
    if which == "alpaca":
        print(f"[data] source=alpaca, fetching {len(syms)} symbols {start} -> {end}")
        return AlpacaSource().fetch(syms, start, end, batch=100, pause=0.15,
                                    cache_dir=None)

    # Only ask Yahoo for names that still exist. Fetching ~1,900 dead tickers
    # cost roughly half the runtime and returned nothing but "possibly
    # delisted" warnings, while making the coverage floor unreachable.
    syms, _ = tradable_universe(syms)
    print(f"[data] source={which}, fetching {len(syms)} symbols {start} -> {end}")
    panel = _fetch_yahoo_with_retry(syms, start, end)
    _assert_coverage(panel, syms)
    return panel


def tradable_universe(syms):
    """Drop symbols the broker cannot actually trade today.

    data/alpaca_universe2.csv is survivorship-inclusive: of its 3,998 tickers,
    roughly 1,906 are companies that were acquired, merged or liquidated --
    AABA, ABMD, XLNX, WORK, ZNGA and so on. That is correct and deliberate for a
    BACKTEST, where a dead company's final months are exactly the data a
    survivorship-free panel must contain.

    It is wrong for live trading, and it silently broke the coverage guard.
    Yahoo is survivor-only, so it will never return a bar for a company that no
    longer exists; measured against the raw file, coverage tops out near 52%.
    The 95% floor was therefore unsatisfiable from the moment it was added on
    2026-08-25, and refused every run thereafter.

    Excluding dead names changes nothing about what the strategy can pick.
    eligible() requires close.notna() (strategy.py:92), so a ticker with no
    price was never rankable and never occupied one of the 490 universe slots.
    This corrects the denominator; it does not lower the bar. Lowering the
    threshold to make 52% "pass" would be the opposite -- it would wave through
    a genuinely truncated fetch, which is the exact failure the guard exists to
    stop.

    Returns (symbols, filtered). On failure it returns the raw list unfiltered,
    so the coverage guard then refuses the run: fail closed, never fail open.
    """
    try:
        assets = api("/v2/assets?status=active&asset_class=us_equity")
    except Exception as exc:
        print(f"[universe] could not reach Alpaca assets ({type(exc).__name__}: "
              f"{exc}); using the raw list. Coverage will be measured against "
              f"delisted names too, so the run will likely refuse.", flush=True)
        return list(syms), False

    ok = {a["symbol"] for a in assets if a.get("tradable")}
    live = [s for s in syms if s in ok]
    dropped = len(syms) - len(live)
    print(f"[universe] {len(live)}/{len(syms)} symbols are tradable today "
          f"({dropped} delisted or untradable, excluded)", flush=True)
    if not live:
        raise RuntimeError(
            "no symbol in the universe file is tradable according to Alpaca; "
            "refusing to trade on an empty universe")
    return live, True


def _covered(panel, syms) -> set:
    """Symbols with a usable price series in the panel."""
    if panel is None or panel.close.empty:
        return set()
    tail = panel.close.tail(5)
    return {s for s in syms if s in tail.columns and tail[s].notna().any()}


def _merge_panels(base, extra):
    """Fill base's gaps from extra. Column-wise, so a retry only adds names."""
    if base is None:
        return extra
    if extra is None or extra.close.empty:
        return base
    j = lambda a, b: a.combine_first(b) if not a.empty else b
    return Panel(j(base.open, extra.open), j(base.high, extra.high),
                 j(base.low, extra.low), j(base.close, extra.close),
                 j(base.volume, extra.volume), j(base.adj_close, extra.adj_close),
                 survivorship_free=base.survivorship_free, source=base.source)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _yahoo_download(syms, start, end, batch, pause, threads):
    """Yahoo fetch with the request rate under our control.

    This deliberately does NOT call YahooSource.fetch. That method hardcodes
    `threads=True` (core/data.py), and parallel requests are precisely what
    Yahoo throttles -- coverage fell 30% -> 51.5% -> 23.0% on consecutive days
    while every other input stayed the same. Serialising the requests is the
    single biggest lever we have, and it is not expressible through that API.

    Kept here rather than patched into core/data.py for the same reason the
    retry wrapper below lives here: the upstream package stays mergeable, and
    deployment concerns stay in the deployment layer.

    Mirrors YahooSource.fetch's Panel assembly exactly so the strategy sees an
    identical structure; only the request pacing differs.
    """
    import warnings

    import yfinance as yf

    warnings.filterwarnings("ignore")
    keys = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    frames = {k: [] for k in keys}
    syms = list(dict.fromkeys(syms))

    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        try:
            raw = yf.download(chunk, start=start, end=end, auto_adjust=False,
                              progress=False, threads=threads, group_by="column")
        except Exception as exc:
            print(f"  batch {i // batch}: FAILED {type(exc).__name__}", flush=True)
            raw = None
        if raw is not None and len(raw):
            top = raw.columns.get_level_values(0)
            for k in keys:
                if k in top:
                    frames[k].append(raw[k])
        print(f"  fetched {min(i + batch, len(syms))}/{len(syms)}", flush=True)
        if i + batch < len(syms):
            time.sleep(pause)

    def merge(key):
        if not frames[key]:
            return pd.DataFrame()
        out = pd.concat(frames[key], axis=1).sort_index()
        return out.loc[:, ~out.columns.duplicated()]

    o, h, l, c, v, a = (merge(k) for k in keys)
    if c.empty:
        raise RuntimeError("Yahoo returned no data")
    cols = c.columns
    align = lambda d: (d.reindex(columns=cols) if not d.empty
                       else pd.DataFrame(index=c.index, columns=cols, dtype=float))
    return Panel(align(o), align(h), align(l), c, align(v), align(a),
                 survivorship_free=False, source="yahoo")


def _fetch_yahoo_with_retry(syms, start, end, attempts=None):
    """Yahoo rate-limits whole batches, not individual symbols.

    yfinance downloads in blocks, so one YFRateLimitError silently drops every
    name in that block -- observed live as a contiguous alphabetical run
    (OKTA..VRSK) vanishing from the panel, which both stranded exits and shrank
    the universe the signal ranks over. Retrying only the missing symbols, with
    a growing pause, recovers them without re-requesting the whole universe.

    Every knob is env-tunable so the coverage probe workflow can search for
    settings that actually clear MIN_COVERAGE without a commit per attempt.
    Defaults are the gentle ones: serial requests, a real pause between batches,
    and a backoff long enough to outlast a throttle window (~30s to ~8 min).
    """
    attempts = attempts if attempts is not None else _env_int("YF_ATTEMPTS", 5)
    batch = _env_int("YF_BATCH", 100)
    pause = _env_float("YF_PAUSE", 2.0)
    threads = os.environ.get("YF_THREADS", "false").strip().lower() == "true"
    # Same floor _assert_coverage enforces: there is no reason to keep asking
    # once the answer is already good enough to trade on.
    floor = _env_float("MIN_COVERAGE", 0.95)
    print(f"[data] fetch settings: batch={batch} pause={pause}s "
          f"threads={threads} attempts={attempts}", flush=True)

    panel, missing = None, list(syms)
    for i in range(1, attempts + 1):
        if i > 1:
            wait = _env_float("YF_BACKOFF", 30.0) * (2 ** (i - 2))
            print(f"[data] {len(missing)} symbols missing; retry {i}/{attempts} "
                  f"after {wait:.0f}s", flush=True)
            time.sleep(wait)
        try:
            got = _yahoo_download(missing, start, end, batch=batch,
                                  pause=pause * i, threads=threads)
        except Exception as exc:
            print(f"[data] attempt {i} failed: {type(exc).__name__}: {exc}", flush=True)
            got = None
        panel = _merge_panels(panel, got)
        have = _covered(panel, syms)
        missing = [s for s in syms if s not in have]
        frac = len(have) / len(syms) if syms else 0.0
        print(f"[data] coverage {len(have)}/{len(syms)} ({frac:.1%})", flush=True)
        if not missing:
            break
        # Stop once the run is going to be allowed to trade. Retrying to 100%
        # is chasing a target that does not exist: a handful of names are
        # simply absent from Yahoo on any given day, and the probe burned 7.5
        # of its 18 minutes on four rounds that recovered none of the last 15
        # symbols. Past the floor, more retries buy nothing and only add load.
        if frac >= floor:
            print(f"[data] above the {floor:.0%} floor; not retrying for the "
                  f"remaining {len(missing)}", flush=True)
            break
    return panel


def _assert_coverage(panel, syms) -> None:
    """Refuse to trade on a truncated universe.

    The strategy ranks the whole universe and takes the top N, so a missing
    block does not merely skip those names -- it changes which names win. A
    thin panel is worse than a missed day, and a missed day is itself costly
    (sleeves stop rolling), so the bar is set high rather than absolute.
    """
    floor = float(os.environ.get("MIN_COVERAGE", "0.95"))
    have = _covered(panel, syms)
    frac = len(have) / len(syms) if syms else 0.0
    if frac < floor:
        raise RuntimeError(
            f"universe coverage {frac:.1%} ({len(have)}/{len(syms)}) is below "
            f"MIN_COVERAGE={floor:.0%}. Refusing to trade on a truncated "
            f"universe — the signal ranks across all names, so gaps change "
            f"which names are selected, not just which are skipped. "
            f"(Denominator is the broker's tradable list, not the raw universe "
            f"file, which carries delisted names Yahoo can never return.)"
        )
    print(f"[data] coverage {frac:.1%} — above the {floor:.0%} floor", flush=True)


def build_plan(equity: float) -> pd.DataFrame:
    panel = fetch_panel()
    tgt = todays_target(panel.close.astype(float),
                        panel.volume.astype(float),
                        panel.open.astype(float))
    if tgt.empty:
        sys.exit("[plan] strategy produced no target positions today")
    held = positions()
    held_qty = position_qtys()
    last = panel.close.astype(float).iloc[-1]

    rows = []
    for sym in sorted(set(tgt.index) | set(held)):
        w = float(tgt.get(sym, 0.0))
        want = w * equity
        have = float(held.get(sym, 0.0))
        px = float(last.get(sym, np.nan))
        rows.append({"symbol": sym, "weight": w, "target_$": want,
                     "current_$": have, "delta_$": want - have, "price": px,
                     "held_qty": float(held_qty.get(sym, 0.0))})
    df = pd.DataFrame(rows)
    # Skip trades too small to be worth the spread.
    df["act"] = df["delta_$"].abs() > max(20.0, 0.002 * equity)
    return df.sort_values("target_$", ascending=False)


def submit(df: pd.DataFrame):
    """Returns (accepted_count, [(symbol, reason), ...]) so callers can report
    rejections. Previously this only printed them, which meant a failed exit
    was invisible outside the CI log."""
    sent = 0
    failed: list[tuple[str, str]] = []
    for _, r in df[df["act"]].iterrows():
        side = "buy" if r["delta_$"] > 0 else "sell"
        have_qty = float(r.get("held_qty", 0.0) or 0.0)
        priced = bool(np.isfinite(r["price"]) and r["price"] > 0)

        # A FULL exit needs no price. qty is the held share count, and the
        # decision to act came from delta_$, which is computed against the
        # broker's own market_value (positions(), line 68) -- not from the data
        # feed. Requiring a price here is what stranded positions permanently:
        # a held name absent from that day's Yahoo panel got price=NaN and was
        # skipped on every subsequent run, so the strategy could never exit it.
        # That is the opposite of the design, where a name leaves the book the
        # day the signal drops it.
        full_exit = side == "sell" and r["weight"] <= 0 and have_qty > 0
        if not priced and not full_exit:
            print(f"  skip {r['symbol']}: no price")
            continue

        # Sizing still needs a price; a full exit sizes from shares instead.
        qty = round(abs(r["delta_$"]) / r["price"], 4) if priced else 0.0

        # A sell size derived from dollars can exceed the shares actually held.
        # delta_$ is the position's market value at the BROKER's live price,
        # while r["price"] is the last close from the data feed. When a name
        # rises overnight the division returns more shares than exist and the
        # broker rejects the order outright -- observed as HTTP 403 on three of
        # four exits in the first live run, each overshooting by 1-3%.
        #
        # So: never offer more shares than are held, and make a full exit
        # (target weight 0) sell exactly the held quantity rather than an
        # estimate of it.
        if side == "sell":
            if have_qty <= 0:
                print(f"  skip {r['symbol']}: nothing held to sell")
                continue
            qty = have_qty if r["weight"] <= 0 else min(qty, have_qty)
            qty = round(qty, 9)
            if not priced:
                print(f"  {r['symbol']}: no price in the feed; "
                      f"exiting on the broker's share count instead")

        if qty <= 0:
            continue
        order = {"symbol": r["symbol"], "qty": str(qty),
                 "side": side,
                 "type": "market", "time_in_force": "day"}
        try:
            o = api("/v2/orders", "POST", order)
            print(f"  SENT {order['side']:<4} {qty:>10} {r['symbol']:<6} id={o.get('id','?')[:8]}")
            sent += 1
        except Exception as exc:
            print(f"  FAILED {r['symbol']}: {exc}")
            failed.append((str(r["symbol"]), str(exc)))
    print(f"[orders] {sent} submitted"
          + (f", {len(failed)} rejected" if failed else ""))
    return sent, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="actually place the orders (default is a dry run)")
    a = ap.parse_args()

    eq, cash, status = account()
    print("=" * 68)
    print(f"ALPACA PAPER ACCOUNT   status={status}   equity=${eq:,.2f}   cash=${cash:,.2f}")
    print(f"{NAME}: {DESCRIPTION}")
    print("=" * 68)

    plan = build_plan(eq)
    show = plan[(plan["weight"] > 0) | (plan["current_$"].abs() > 1)]
    print(f"\n{'symbol':<8}{'weight':>9}{'target $':>12}{'current $':>12}{'delta $':>12}  act")
    print("-" * 68)
    for _, r in show.iterrows():
        print(f"{r['symbol']:<8}{r['weight']:>8.2%}{r['target_$']:>12,.2f}"
              f"{r['current_$']:>12,.2f}{r['delta_$']:>+12,.2f}  {'YES' if r['act'] else '-'}")
    n = int(plan["act"].sum())
    turn = plan.loc[plan["act"], "delta_$"].abs().sum()
    print("-" * 68)
    print(f"{n} orders, ${turn:,.2f} traded ({turn / eq:.1%} of equity)")

    if not a.execute:
        print("\nDRY RUN -- nothing was sent.")
        print("Review the plan above, then re-run with --execute to place these orders.")
        return
    print("\nplacing orders...")
    submit(plan)


if __name__ == "__main__":
    main()
