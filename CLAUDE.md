# botting-momentum — agent instructions

This repo runs the **momentum** strategy (`strategy.py`) against an Alpaca
account. It ran on paper (PA3HT4JN9ZQY) from 2026-09-29, and was moved to the
owner's **live, real-money** account at their explicit instruction on
2026-10-06. The paper record is kept in `state/history_paper.csv`. Equities only; never query, analyze or
trade cryptocurrency.

## Rules

1. **Paper by default.** `paper_trade.resolve_base()` returns the paper endpoint
   unless `ALPACA_LIVE` is exactly `true` AND `ALPACA_ACCOUNT_ID` and
   `MAX_DEPLOY` are set; the API key prefix must match (PK = paper, AK = live).
   Do not relax this. `tests/test_live_guard.py` pins it.
2. **One bot per account.** The live account was the assay bot's
   (OMV0601/botting-it-up), which is halted and must stay off it: two bots
   on one account would each sell the other's positions.
3. **Its own account only.** An executing run requires `ALPACA_ACCOUNT_ID`, and
   `paper_trade.assert_expected_account()` refuses any other account, including
   the assay bot's (`OTHER_BOTS_ACCOUNTS`).
4. **Long-only, unlevered, no shorting.** `run_daily.py` asserts gross <= 1.0
   and no negative weights before sending anything.
5. **No stop-losses by design.** Exits come from the strategy rule itself
   during the daily run. Don't bolt stops on; that changes what's being tested.
6. **The daily run matters.** A missed day leaves the book off-target.
7. **Halting** needs no code change: `HALT` file in the repo root, or the
   `TRADING_ENABLED` variable set to `false`.
8. **Log every action** to `journal.md` (`run_daily.py` already does).

## The strategy

`strategy.py` is the only place the algorithm lives. It must expose `NAME`,
`DESCRIPTION`, `MIN_PRICE`, `MIN_DOLLAR_VOLUME`, `LONGEST_LOOKBACK`,
`target_weights(close, volume, open_)` and `todays_target(close, volume, open_)`.
Parameters were taken from the paper, not searched; don't tune them to make a
backtest look better.

## Code standards

- Live-path files: `strategy.py`, `paper_trade.py`, `run_daily.py`,
  `preflight.py`, `liquidate.py`. Run `python -m pytest tests/ -q` before pushing.
- Never commit credentials; keys live in GitHub Actions secrets
  `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY`.
- `data/` is a gitignored cache except `data/alpaca_universe2.csv`, which the
  live path reads every run.
