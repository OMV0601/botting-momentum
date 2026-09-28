"""
Nothing reaches real money without three deliberate settings.

Placing an order against api.alpaca.markets spends money that cannot be
recalled. The guard is therefore not one check but a set, each covering a
different way the switch could be made by accident rather than on purpose:

  ALPACA_LIVE=true    the intent. Exact string only -- "1", "yes", "TRUE" and
                      a stray space all stay on paper.
  ALPACA_ACCOUNT_ID   which account, so a live run cannot start unpinned.
  MAX_DEPLOY          how much, because with no cap the strategy deploys 100%
                      of whatever balance the credentials happen to reach.

Plus the key/endpoint agreement: Alpaca issues paper keys as PK... and live
keys as AK..., so a mismatch means a half-finished switch.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trade as pt

LIVE_VARS = ("ALPACA_LIVE", "ALPACA_ACCOUNT_ID", "MAX_DEPLOY",
             "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in LIVE_VARS:
        monkeypatch.delenv(v, raising=False)


def test_default_is_paper():
    assert pt.resolve_base() == pt.PAPER_BASE


@pytest.mark.parametrize("value", ["", "false", "False", "0", "1", "yes",
                                   "TRUE", "True", " true", "true "])
def test_only_the_exact_string_true_opts_in(value, monkeypatch):
    """Real money must not be reachable by a typo or a shell quirk."""
    monkeypatch.setenv("ALPACA_LIVE", value)
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "ACCT")
    monkeypatch.setenv("MAX_DEPLOY", "5000")
    assert pt.resolve_base() == pt.PAPER_BASE


def test_live_needs_both_pin_and_cap(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE", "true")
    with pytest.raises(RuntimeError, match="ALPACA_ACCOUNT_ID and MAX_DEPLOY"):
        pt.resolve_base()

    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "ACCT")
    with pytest.raises(RuntimeError, match="MAX_DEPLOY"):
        pt.resolve_base()

    monkeypatch.setenv("MAX_DEPLOY", "5000")
    assert pt.resolve_base() == pt.LIVE_BASE


def test_blank_pin_or_cap_does_not_count(monkeypatch):
    """An Actions variable that exists but is empty is the classic way this
    goes wrong -- ${{ vars.MAX_DEPLOY }} with nothing behind it."""
    monkeypatch.setenv("ALPACA_LIVE", "true")
    monkeypatch.setenv("ALPACA_ACCOUNT_ID", "   ")
    monkeypatch.setenv("MAX_DEPLOY", "")
    with pytest.raises(RuntimeError):
        pt.resolve_base()


# --- key / endpoint agreement ---------------------------------------------

def test_live_key_on_paper_endpoint_is_refused():
    """The dangerous direction: real credentials loaded but the URL not
    switched. Paper keys on the live endpoint merely fail to authenticate;
    this one could look like it is working."""
    with pytest.raises(RuntimeError, match="LIVE API key"):
        pt.assert_key_matches_endpoint(pt.PAPER_BASE, "AKFAKE123")


def test_paper_key_on_live_endpoint_is_refused():
    with pytest.raises(RuntimeError, match="non-live API key"):
        pt.assert_key_matches_endpoint(pt.LIVE_BASE, "PKFAKE123")


@pytest.mark.parametrize("base,key", [(pt.PAPER_BASE, "PKFAKE"),
                                      (pt.LIVE_BASE, "AKFAKE")])
def test_matching_pairs_pass(base, key):
    pt.assert_key_matches_endpoint(base, key)


# --- the api() call itself -------------------------------------------------

def test_api_refuses_an_unknown_endpoint():
    with pytest.raises(AssertionError, match="unknown trading endpoint"):
        pt.api("/v2/account", base="https://evil.example.com")


def test_api_refuses_live_base_without_the_opt_in(monkeypatch):
    """Passing LIVE_BASE explicitly must not bypass ALPACA_LIVE."""
    monkeypatch.setenv("ALPACA_API_KEY_ID", "AKFAKE")
    with pytest.raises(RuntimeError, match="without ALPACA_LIVE"):
        pt.api("/v2/account", base=pt.LIVE_BASE)


def test_preflight_agrees_with_paper_trade(monkeypatch):
    """preflight duplicates resolve_base rather than importing paper_trade,
    because it must still run when numpy/pandas are missing. Duplication is
    only safe while the two actually agree."""
    import preflight

    for env, expect in [
        ({}, pt.PAPER_BASE),
        ({"ALPACA_LIVE": "false"}, pt.PAPER_BASE),
        ({"ALPACA_LIVE": "true", "ALPACA_ACCOUNT_ID": "A",
          "MAX_DEPLOY": "5000"}, pt.LIVE_BASE),
    ]:
        for v in LIVE_VARS:
            monkeypatch.delenv(v, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert preflight._base() == pt.resolve_base() == expect
