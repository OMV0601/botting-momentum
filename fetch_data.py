"""
Pull the raw panel.

Ticker selection is deliberately NOT filtered on current price. Choosing which
symbols to download based on what they cost today would bake a selection bias
straight into the universe before the backtest even starts -- a stock that
trades at $12 today may have been $400 in 2021. The point-in-time filter in
the engine does the $5-20 selection, using only the prior close.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import get_source, list_us_symbols

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-08-18")
    ap.add_argument("--n-tickers", type=int, default=2500)
    ap.add_argument("--source", default="auto")
    ap.add_argument("--out", default="data/panel")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src = get_source(a.source)

    if hasattr(src, "list_symbols"):
        syms = src.list_symbols(include_inactive=True)
        print(f"[universe] {len(syms)} symbols "
              f"({(syms['status'] == 'inactive').sum()} delisted included)")
        tickers = syms["Symbol"].tolist()
    else:
        df = list_us_symbols()
        tickers = df["Symbol"].tolist()
        print(f"[universe] {len(tickers)} currently-listed common stocks "
              f"(NO delisted names available from this source)")

    # Unbiased subsample: random, seeded, never price-based.
    if a.n_tickers and len(tickers) > a.n_tickers:
        rng = np.random.default_rng(a.seed)
        tickers = sorted(rng.choice(tickers, a.n_tickers, replace=False).tolist())
        print(f"[universe] random seeded subsample -> {len(tickers)}")

    print(f"[fetch] {a.start} -> {a.end}")
    panel = src.fetch(tickers, a.start, a.end)

    # Drop names with almost no history; they only add NaN noise.
    keep = panel.close.notna().sum() >= 250
    for fld in ["open", "high", "low", "close", "volume", "adj_close"]:
        setattr(panel, fld, getattr(panel, fld).loc[:, keep[keep].index])
    print(f"[fetch] kept {keep.sum()} tickers with >=250 days")
    print("[fetch]", panel.summary())

    panel.save(a.out)
    print(f"[fetch] saved -> {a.out}")
