"""
Email notifications via Resend.

Sends one summary per executed rebalance, plus an alert if a run fails.

DESIGN RULE: notification failure must never affect trading. Every public
function here swallows its own exceptions and returns a bool. A dead API key,
a network blip or a Resend outage degrades to "no email", never to a missed or
partial rebalance.

Configuration:
    RESEND_API_KEY   required; without it every send is skipped quietly
    NOTIFY_TO        recipient, defaults to DEFAULT_TO below
    NOTIFY_FROM      sender, defaults to Resend's shared onboarding address

Resend's shared `onboarding@resend.dev` sender needs no domain verification but
will only deliver to the address the Resend account was registered with. To
send anywhere else, verify a domain and set NOTIFY_FROM.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

API = "https://api.resend.com/emails"
DEFAULT_TO = "omvyas.0601@gmail.com"
try:
    from strategy import NAME as BOT_NAME
except Exception:  # notifications must never fail on import
    BOT_NAME = "trading"
DEFAULT_FROM = f"{BOT_NAME} bot <onboarding@resend.dev>"

GREEN, RED, GREY = "#1a7f37", "#c0392b", "#6b7280"


def _cfg():
    key = os.environ.get("RESEND_API_KEY", "").strip()
    to = os.environ.get("NOTIFY_TO", "").strip() or DEFAULT_TO
    frm = os.environ.get("NOTIFY_FROM", "").strip() or DEFAULT_FROM
    return key, to, frm


def send(subject: str, html: str) -> bool:
    """Return True if Resend accepted the message. Never raises."""
    key, to, frm = _cfg()
    if not key:
        print("[notify] RESEND_API_KEY not set — skipping email", flush=True)
        return False
    body = json.dumps({"from": frm, "to": [to], "subject": subject,
                       "html": html}).encode()
    # The User-Agent is load-bearing. Resend sits behind Cloudflare, which
    # rejects urllib's default "Python-urllib/3.x" signature with HTTP 403 and
    # "error code: 1010" (banned browser signature) before the request ever
    # reaches Resend -- observed on the first live send.
    req = urllib.request.Request(
        API, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Accept": "application/json",
                 "User-Agent": f"{BOT_NAME}-bot/1.0 (+github-actions)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            mid = json.loads(r.read().decode()).get("id", "?")
        print(f"[notify] sent to {to} (id={mid})", flush=True)
        return True
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:300]
        except Exception:
            pass
        print(f"[notify] FAILED HTTP {exc.code}: {detail}", flush=True)
    except Exception as exc:
        print(f"[notify] FAILED {type(exc).__name__}: {exc}", flush=True)
    return False


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _rows(orders) -> str:
    if not orders:
        return f'<tr><td colspan="4" style="padding:10px;color:{GREY}">no orders</td></tr>'
    out = []
    for o in orders:
        buy = o["delta"] > 0
        colour = GREEN if buy else RED
        out.append(
            f'<tr>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:600;color:{colour}">'
            f'{"BUY" if buy else "SELL"}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;font-family:monospace">{o["symbol"]}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right">{o["weight"]:.2%}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right;font-family:monospace">'
            f'{o["delta"]:+,.2f}</td>'
            f'</tr>')
    return "".join(out)


def daily_summary(*, equity: float, deployed: float, orders: list,
                  sent: int, failed: list, prev_equity: float | None = None,
                  dry_run: bool = False, session: dict | None = None,
                  since_date: str | None = None) -> bool:
    """orders: [{symbol, weight, delta}]  failed: [(symbol, reason)]

    The headline number is the day so far -- market open to right now, from
    Alpaca's own minute-by-minute record.

    It used to be `equity now - equity when the bot last ran`, shown bare with
    no span label at all. On 2026-09-03 that read "+69.74 (+1.39% of the
    $5,000 book)" for a span running from 14:28 UTC the previous day: almost
    25 hours, most of it overnight while the market was shut. Nothing on the
    email said so, so it read as "what the bot made today". It was not.

    That baseline also drifts with the schedule -- a rebalance at 13:30 and one
    at 17:08 produce numbers measured over different spans -- so the series
    cannot be compared day to day. The close-of-day email was fixed first; this
    is the same fix for the morning one.

    When Alpaca's session record is unavailable the older figure is still
    shown, but labelled for exactly what it is rather than left bare.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    delta_txt = ""
    if session and session.get("n_points", 0) >= 2:
        d = session["pnl"]
        colour = GREEN if d >= 0 else RED
        pct_of_book = f" ({d / deployed:+.2%} of the ${deployed:,.0f} book)" if deployed else ""
        delta_txt = (
            f'<p style="margin:4px 0;font-size:22px;font-weight:700;color:{colour}">'
            f'{d:+,.2f}<span style="font-size:13px;font-weight:400;color:{GREY}">'
            f'{pct_of_book}</span></p>'
            f'<p style="margin:0 0 6px;color:{GREY};font-size:12px">'
            f'today so far — market open to now, still trading</p>'
            f'<p style="margin:0 0 14px;color:{GREY};font-size:11px;font-family:monospace">'
            f'{_money(session["open_equity"])} at the open → '
            f'{_money(session["close_equity"])} now</p>')
    elif prev_equity:
        # Fallback. Name the span instead of leaving a bare number that reads
        # as today's profit -- this is the exact misreading being fixed.
        d = equity - prev_equity
        colour = GREEN if d >= 0 else RED
        since = f" on {since_date}" if since_date else ""
        delta_txt = (
            f'<p style="margin:4px 0;font-size:20px;font-weight:700;color:{colour}">'
            f'{d:+,.2f}</p>'
            f'<p style="margin:0 0 14px;color:#b45309;font-size:12px">'
            f'Alpaca\'s session record was unavailable, so this is measured from '
            f'the last completed run{since} — <b>not</b> today\'s P&amp;L.</p>')

    fail_block = ""
    if failed:
        items = "".join(
            f'<li style="margin:3px 0"><b style="font-family:monospace">{s}</b> — {r}</li>'
            for s, r in failed)
        fail_block = (
            f'<div style="margin:18px 0;padding:12px 14px;background:#fdf2f2;'
            f'border-left:3px solid {RED};border-radius:3px">'
            f'<b style="color:{RED}">{len(failed)} order(s) rejected</b>'
            f'<ul style="margin:8px 0 0;padding-left:20px;font-size:13px">{items}</ul></div>')

    tag = ' <span style="color:#b45309">[DRY RUN — nothing sent]</span>' if dry_run else ""

    html = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
max-width:640px;margin:0 auto;padding:24px;color:#111">
  <h2 style="margin:0 0 2px;font-size:17px">{BOT_NAME} daily rebalance{tag}</h2>
  <p style="margin:0 0 18px;color:{GREY};font-size:12px">{stamp}</p>
  {delta_txt}
  <table style="width:100%;border-collapse:collapse;margin:16px 0;font-size:13px">
    <tr><td style="padding:5px 0;color:{GREY}">Account equity</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{_money(equity)}</td></tr>
    <tr><td style="padding:5px 0;color:{GREY}">Deployed</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{_money(deployed)}</td></tr>
    <tr><td style="padding:5px 0;color:{GREY}">Orders accepted</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{sent}</td></tr>
  </table>
  {fail_block}
  <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:12px">
    <tr style="text-align:left;color:{GREY};font-size:11px;text-transform:uppercase">
      <th style="padding:0 10px 6px">Side</th><th style="padding:0 10px 6px">Symbol</th>
      <th style="padding:0 10px 6px;text-align:right">Weight</th>
      <th style="padding:0 10px 6px;text-align:right">Delta $</th></tr>
    {_rows(orders)}
  </table>
  <p style="margin-top:22px;color:{GREY};font-size:11px;line-height:1.5">
    Dedicated Alpaca paper account. The whole balance is deployed to this
    strategy, so the account's own move IS the strategy's return — there is no
    dilution factor left to correct for.</p>
</div>"""

    n = len(orders)
    subject = f"{BOT_NAME} — {sent} order{'s' if sent != 1 else ''}"
    if failed:
        subject += f", {len(failed)} rejected"
    if session and session.get("n_points", 0) >= 2:
        subject += f" — {session['pnl']:+,.2f} today"
    elif prev_equity:
        subject += f" — {equity - prev_equity:+,.2f} (since last run)"
    if dry_run:
        subject = "[dry run] " + subject
    return send(subject, html)


def close_summary(*, equity: float, cash: float, position_value: float,
                  n_positions: int, deployed: float,
                  open_equity: float | None = None, traded_today: bool = False,
                  orders_sent: int = 0, rejected: int = 0,
                  since_date: str | None = None,
                  session: dict | None = None) -> bool:
    """End-of-day P&L, sent after the close.

    Separate from daily_summary because it answers a different question. The
    morning email says what was traded; this says how the day went, with prices
    settled. It is also the one that proves the bot is alive on a day where the
    rebalance had nothing to do.

    Leads with dollars rather than an account percentage. That began as a
    correction for MAX_DEPLOY: only part of the account was at work, so the
    account-level percentage understated the strategy by the dilution factor.
    Since the 2026-09-05 move to a dedicated account the two agree, and dollars
    stay the headline for the plainer reason that they are what a $5,000 book
    is legible in.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # THE headline: open to close, from Alpaca's own minute-by-minute record.
    # Anything measured from "when the bot last ran" drifts with the schedule
    # and is not comparable across days.
    change = ""
    if session and session.get("n_points", 0) >= 2:
        d = session["pnl"]
        colour = GREEN if d >= 0 else RED
        pct = f" ({d / deployed:+.2%} of the ${deployed:,.0f} book)" if deployed else ""
        change = (
            f'<p style="margin:4px 0;font-size:26px;font-weight:700;color:{colour}">'
            f'{d:+,.2f}<span style="font-size:13px;font-weight:400;color:{GREY}">'
            f'{pct}</span></p>'
            f'<p style="margin:0 0 6px;color:{GREY};font-size:12px">'
            f'market open to close — the day\'s actual trading</p>'
            f'<p style="margin:0 0 14px;color:{GREY};font-size:11px;font-family:monospace">'
            f'{_money(session["open_equity"])} at the open → '
            f'{_money(session["close_equity"])} at the close</p>')
    elif open_equity:
        # Fallback only. Say plainly that this is the weaker measure rather
        # than dressing it up as the day's result.
        d = equity - open_equity
        colour = GREEN if d >= 0 else RED
        since = f" on {since_date}" if since_date else ""
        change = (
            f'<p style="margin:4px 0;font-size:22px;font-weight:700;color:{colour}">'
            f'{d:+,.2f}</p>'
            f'<p style="margin:0 0 14px;color:#b45309;font-size:12px">'
            f'Alpaca\'s session record was unavailable, so this is measured from '
            f'the last completed run{since} — <b>not</b> an open-to-close figure.</p>')

    # An over-sized book is the visible symptom of exits that did not complete,
    # which is precisely how the book drifted to ~$6,546 against a $5,000 cap.
    # Flag it here rather than leaving it to be noticed in a screenshot.
    drift = ""
    if deployed and position_value > deployed * 1.15:
        drift = (f'<div style="margin:16px 0;padding:12px 14px;background:#fffbeb;'
                 f'border-left:3px solid #b45309;border-radius:3px;font-size:13px">'
                 f'<b style="color:#b45309">Book is larger than the cap</b><br>'
                 f'Holding {_money(position_value)} against a {_money(deployed)} '
                 f'target ({position_value / deployed:.0%}). Usually means exits '
                 f'are not completing.</div>')

    traded = (f'{orders_sent} order{"s" if orders_sent != 1 else ""} this morning'
              + (f', {rejected} rejected' if rejected else '')) if traded_today \
        else '<span style="color:#b45309">no rebalance ran today</span>'

    html = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
max-width:640px;margin:0 auto;padding:24px;color:#111">
  <h2 style="margin:0 0 2px;font-size:17px">{BOT_NAME} — close of day</h2>
  <p style="margin:0 0 18px;color:{GREY};font-size:12px">{stamp}</p>
  {change}
  {drift}
  <table style="width:100%;border-collapse:collapse;margin:16px 0;font-size:13px">
    <tr><td style="padding:5px 0;color:{GREY}">Account equity</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{_money(equity)}</td></tr>
    <tr><td style="padding:5px 0;color:{GREY}">In positions</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{_money(position_value)}</td></tr>
    <tr><td style="padding:5px 0;color:{GREY}">Cash</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{_money(cash)}</td></tr>
    <tr><td style="padding:5px 0;color:{GREY}">Target deployed</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{_money(deployed)}</td></tr>
    <tr><td style="padding:5px 0;color:{GREY}">Names held</td>
        <td style="padding:5px 0;text-align:right;font-family:monospace">{n_positions}</td></tr>
    <tr><td style="padding:5px 0;color:{GREY}">Today</td>
        <td style="padding:5px 0;text-align:right">{traded}</td></tr>
  </table>
  <p style="margin-top:22px;color:{GREY};font-size:11px;line-height:1.5">
    Dedicated Alpaca paper account. The whole balance is deployed to this
    strategy, so the account's own move IS the strategy's return — there is no
    dilution factor left to correct for.</p>
</div>"""

    subject = f"{BOT_NAME} — close of day"
    if session and session.get("n_points", 0) >= 2:
        subject += f" — {session['pnl']:+,.2f}"
    elif open_equity:
        subject += f" — {equity - open_equity:+,.2f} (since last run)"
    if not traded_today:
        subject += " (no rebalance)"
    return send(subject, html)


def failure_alert(stage: str, detail: str) -> bool:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
max-width:640px;margin:0 auto;padding:24px;color:#111">
  <h2 style="margin:0 0 2px;font-size:17px;color:{RED}">{BOT_NAME} run FAILED</h2>
  <p style="margin:0 0 16px;color:{GREY};font-size:12px">{stamp} — {stage}</p>
  <pre style="background:#f6f8fa;padding:12px;border-radius:4px;font-size:12px;
white-space:pre-wrap;word-break:break-word">{detail[:3000]}</pre>
  <p style="color:{GREY};font-size:11px">No rebalance happened. The strategy
  assumes one run per trading day, so a missed day lets the book drift from
  its target.</p>
</div>"""
    return send(f"{BOT_NAME} FAILED — {stage}", html)
