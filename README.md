# botting-momentum

An automated **momentum** strategy for an Alpaca **paper** account, $5,000 starting capital.

12-1 price momentum: buy the top 10% of stocks by their last-12-months return (skipping the latest month), equal weight, each pick held about a month.

> Not investment advice. A backtest is not a promise; results below come from past data.

## The algorithm

- Every trading day, score each stock: **price 21 days ago ÷ price 252 days ago − 1** (last year's return, skipping the latest month).
- Buy the **top 10%** of the universe by that score, equal weight.
- Hold each day's picks for **21 trading days**. The book is 21 overlapping "sleeves", one refreshed each day, so there's no lucky or unlucky rebalance day.
- Sell a stock once it has dropped out of the top 10% for 21 days in a row.

**Universe** (the same as the assay bot, so results compare fairly): US stocks priced at $3 or more, trading at least $5M a day, the 490 most liquid, all measured with a one-day lag. Equities only, no crypto.

**Risk:** long-only, no leverage, no shorting, no stop-losses. The whole account is the strategy.
Momentum can crash hard when markets turn sharply. For example, it lost 23% in 2018. Expect big swings.

## Backtest

Alpaca data including delisted companies, 2017–2026, with trading costs:

| $5,000 → (2017–2026) | yearly return | worst drop | Sharpe | names held |
|---|---|---|---|---|
| see backtest workflow | **13.0%** | **−52%** | 0.54 | ~125 |

Run it yourself from the **backtest** workflow in the Actions tab, or locally:

```bash
pip install -r requirements.txt
python fetch_data.py --source alpaca --start 2016-01-01 --end 2026-09-25 --n-tickers 4000 --out data/panel
python backtest.py --panel data/panel
```

Source: Kakushadze & Serur, *151 Trading Strategies* (SSRN 3247865), Section 3.1.

## How it runs

Once per trading day at the open, GitHub Actions runs `run_daily.py`:

1. Checks the market is open, and that it hasn't already traded today.
2. Downloads prices, computes today's target from `strategy.py`.
3. Checks safety: paper endpoint, right account, no shorts, no leverage.
4. Sends market orders for the difference, then logs to `journal.md` and emails a summary.

| File | What it does |
|---|---|
| `strategy.py` | **The algorithm.** Nothing else decides what to buy. |
| `run_daily.py` | Daily entrypoint with gates, safety checks and logging. |
| `paper_trade.py` | Alpaca API, price fetch, plan and orders. |
| `preflight.py` | Checks keys, account and clock. Places no orders. |
| `liquidate.py` | Manual "sell everything" (needs `LIQUIDATE` typed in). |
| `backtest.py` | Backtest from $5,000. |
| `core/` | Backtest engine, costs, data loaders (from the assay project). |

## Setup (not done yet)

Nothing trades until these are set. Each bot needs **its own** Alpaca paper account; it refuses to run on the assay bot's account.

1. Create a new Alpaca paper account funded with $5,000.
2. Repo **secrets**: `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` (paper keys start with `PK`).
3. Repo **variable**: `ALPACA_ACCOUNT_ID` = that account's number.
4. Run the **preflight** workflow and confirm it's green.
5. Run **daily rebalance** by hand without "execute" (a dry run), and check the plan.
6. Hook up the daily trigger (the same external clock the assay bot uses).

Optional: `RESEND_API_KEY` secret and `NOTIFY_TO` variable for emails.

**Stop trading:** set the variable `TRADING_ENABLED=false`, or add a file named `HALT` to the repo root.
