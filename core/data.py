"""
Data layer.

Design constraint discovered empirically (2026-08): Yahoo/yfinance returns an
EMPTY history for any delisted symbol -- verified against BBBYQ, WEWKQ, RADCQ,
PTRAQ, EXPRQ, CANO, SAVE, LTHM and ATVI. ATVI is instructive: a clean
Microsoft acquisition, not a bankruptcy, and its entire price history is gone.

That means a Yahoo-sourced universe contains only present-day survivors. In a
$5-20 universe this is not a rounding error, it is the dominant bias: the
names that fell out are disproportionately the ones that went to zero.

So the loader is pluggable. YahooSource is the zero-setup default and is
permanently flagged survivorship_free = False. Any result computed on it is
labelled as biased everywhere it surfaces. AlpacaSource retains delisted
symbols and is flagged True.
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"
CACHE.mkdir(exist_ok=True)


@dataclass
class Panel:
    """Aligned OHLCV panel: index=dates, columns=tickers."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    adj_close: pd.DataFrame
    survivorship_free: bool
    source: str
    last_date: pd.Series = field(default_factory=pd.Series)

    def __post_init__(self):
        if self.last_date is None or len(self.last_date) == 0:
            self.last_date = self.close.apply(lambda c: c.last_valid_index())

    @property
    def tickers(self):
        return list(self.close.columns)

    def trim_sparse_rows(self, min_coverage: float = 0.8) -> "Panel":
        """Drop dates where most names have no bar.

        The final row of a same-day fetch is routinely a partial print -- on
        the S&P panel it arrived with 2 of 498 names populated. Left in, it
        turns into a universe-wide fake gap that the engine reads as a mass
        delisting event.
        """
        cov = self.close.notna().sum(axis=1) / max(self.close.shape[1], 1)
        keep = cov >= min_coverage
        if keep.all():
            return self
        for fld in ["open", "high", "low", "close", "volume", "adj_close"]:
            setattr(self, fld, getattr(self, fld).loc[keep])
        return self

    def summary(self) -> str:
        return (
            f"Panel[{self.source}] {self.close.shape[0]} days x "
            f"{self.close.shape[1]} tickers | "
            f"{self.close.index[0].date()} -> {self.close.index[-1].date()} | "
            f"survivorship-free={self.survivorship_free}"
        )

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in ["open", "high", "low", "close", "volume", "adj_close"]:
            getattr(self, name).to_parquet(path / f"{name}.parquet")
        meta = {"survivorship_free": self.survivorship_free, "source": self.source}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        frames = {
            n: pd.read_parquet(path / f"{n}.parquet")
            for n in ["open", "high", "low", "close", "volume", "adj_close"]
        }
        return cls(survivorship_free=meta["survivorship_free"],
                   source=meta["source"], **frames)


def list_us_symbols(include_etf: bool = False) -> pd.DataFrame:
    """Current NASDAQ-traded symbol directory.

    This is a *current* listing, so on its own it is survivor-only. It serves
    as a candidate pool; sources that retain delisted names supplement it.
    """
    cache = CACHE / "nasdaq_traded.csv"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400:
        return pd.read_csv(cache)
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode()
    df = pd.read_csv(io.StringIO(raw), sep="|")
    df = df[df["Nasdaq Traded"] == "Y"]
    df = df[df["Test Issue"] == "N"]
    if not include_etf:
        df = df[df["ETF"] == "N"]

    # Drop warrants, units, rights and preferreds. These are not common stock
    # and their price dynamics are driven by the terms of the instrument.
    bad_suffix = ("W", "U", "R", "P")

    def is_common(sym) -> bool:
        s = str(sym)
        if "." in s or "$" in s:
            return False
        if len(s) == 5 and s[-1] in bad_suffix:
            return False
        return True

    df = df[df["Symbol"].map(is_common)]
    df = df[["Symbol", "Security Name", "Listing Exchange"]].reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


class YahooSource:
    """Zero-setup default. SURVIVOR-ONLY -- see module docstring."""

    survivorship_free = False
    name = "yahoo"

    def fetch(self, tickers, start, end, batch=150, pause=1.0) -> Panel:
        import warnings

        import yfinance as yf

        warnings.filterwarnings("ignore")
        keys = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        frames = {k: [] for k in keys}
        tickers = list(dict.fromkeys(tickers))

        for i in range(0, len(tickers), batch):
            chunk = tickers[i : i + batch]
            try:
                raw = yf.download(
                    chunk, start=start, end=end, auto_adjust=False,
                    progress=False, threads=True, group_by="column",
                )
            except Exception as exc:
                print(f"  batch {i // batch}: FAILED {type(exc).__name__}")
                continue
            if raw is None or len(raw) == 0:
                continue
            top = raw.columns.get_level_values(0)
            for k in keys:
                if k in top:
                    frames[k].append(raw[k])
            print(f"  fetched {min(i + batch, len(tickers))}/{len(tickers)}", flush=True)
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
        align = lambda d: d.reindex(columns=cols) if not d.empty else pd.DataFrame(index=c.index, columns=cols, dtype=float)
        return Panel(align(o), align(h), align(l), c, align(v), align(a),
                     survivorship_free=False, source="yahoo")


class AlpacaSource:
    """Survivorship-free. Requires free keys from alpaca.markets.

    Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY. Alpaca retains bars for
    delisted symbols, and its /v2/assets endpoint with status=inactive gives
    the dead-ticker list needed to rebuild a point-in-time universe.
    """

    survivorship_free = True
    name = "alpaca"
    BARS = "https://data.alpaca.markets/v2/stocks/bars"
    # Paper and live accounts use different trading hosts, and a paper key
    # returns 401 against the live host. Data (data.alpaca.markets) is shared.
    ASSET_HOSTS = ("https://paper-api.alpaca.markets/v2/assets",
                   "https://api.alpaca.markets/v2/assets")

    def __init__(self):
        self._rejected: set[str] = set()
        self.key = os.environ.get("ALPACA_API_KEY_ID")
        self.sec = os.environ.get("ALPACA_API_SECRET_KEY")
        if not (self.key and self.sec):
            raise RuntimeError(
                "AlpacaSource needs ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY"
            )

    def _get(self, url):
        req = urllib.request.Request(
            url, headers={"APCA-API-KEY-ID": self.key,
                          "APCA-API-SECRET-KEY": self.sec}
        )
        return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())

    def _assets_base(self) -> str:
        """Whichever trading host this key is valid against."""
        if getattr(self, "_asset_host", None):
            return self._asset_host
        last = None
        for host in self.ASSET_HOSTS:
            try:
                self._get(f"{host}?status=active&asset_class=us_equity")
                self._asset_host = host
                return host
            except Exception as exc:
                last = exc
        raise RuntimeError(f"no Alpaca assets host accepted these keys: {last}")

    def list_symbols(self, include_inactive=True) -> pd.DataFrame:
        base = self._assets_base()
        rows = []
        statuses = ["active", "inactive"] if include_inactive else ["active"]
        for status in statuses:
            url = f"{base}?status={status}&asset_class=us_equity"
            for a in self._get(url):
                rows.append({"Symbol": a["symbol"], "status": status,
                             "exchange": a.get("exchange"),
                             "Security Name": a.get("name"),
                             "tradable": a.get("tradable")})
        return pd.DataFrame(rows).drop_duplicates("Symbol")

    def _fetch_chunk(self, chunk, start, end, pause, store, depth=0,
                     timeframe="1Day") -> bool:
        """Fetch one symbol list, following pagination. True if it succeeded.

        Alpaca rejects the WHOLE request with a 400 if any single symbol is
        unknown to it -- e.g. "BRK-B", which it spells "BRK.B". One bad ticker
        therefore silently destroys 99 good ones. The error body names the
        offender, so parse it out, drop it, and retry rather than losing the
        chunk or bisecting blindly.
        """
        chunk = list(chunk)
        page = None
        while True:
            q = {"symbols": ",".join(chunk), "timeframe": timeframe,
                 "start": str(start), "end": str(end), "limit": "10000",
                 "adjustment": "all", "feed": "sip"}
            if page:
                q["page_token"] = page
            try:
                js = self._get(self.BARS + "?" + urllib.parse.urlencode(q))
            except urllib.error.HTTPError as exc:
                bad = None
                try:
                    body = exc.read().decode()
                    if "invalid symbol" in body:
                        bad = json.loads(body).get("message","").split("invalid symbol:")[-1].strip()
                except Exception:
                    pass
                if bad and bad in chunk and depth < 12:
                    chunk.remove(bad)
                    self._rejected.add(bad)
                    if not chunk:
                        return True
                    page = None
                    time.sleep(pause)
                    depth += 1
                    continue
                print(f"    chunk of {len(chunk)}: HTTP {exc.code}", flush=True)
                return False
            except Exception as exc:
                print(f"    chunk of {len(chunk)}: {type(exc).__name__}", flush=True)
                return False
            for sym, bars in (js.get("bars") or {}).items():
                store.setdefault(sym, []).extend(bars)
            page = js.get("next_page_token")
            if not page:
                return True
            time.sleep(pause)

    def fetch(self, tickers, start, end, batch=100, pause=0.35,
              cache_dir=None) -> Panel:
        """cache_dir persists each batch as it lands.

        Without it a failure during the final assembly step throws away every
        request made -- which is exactly what happened once, an hour in.
        """
        store: dict[str, list] = {}
        tickers = list(dict.fromkeys(tickers))
        cdir = Path(cache_dir) if cache_dir else None
        if cdir:
            cdir.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(tickers), batch):
            chunk = tickers[i : i + batch]
            cfile = cdir / f"batch_{i:06d}.json" if cdir else None
            if cfile is not None and cfile.exists():
                store.update(json.loads(cfile.read_text()))
                continue
            # Alpaca rejects long symbol lists with a 400 (URL length), so a
            # failed chunk is halved and retried rather than dropped.
            ok = self._fetch_chunk(chunk, start, end, pause, store)
            if not ok and len(chunk) > 1:
                mid = len(chunk) // 2
                ok = self._fetch_chunk(chunk[:mid], start, end, pause, store)
                ok = self._fetch_chunk(chunk[mid:], start, end, pause, store) and ok
            # Only cache a chunk that actually succeeded. Caching a failure
            # writes an empty file that then permanently masks the gap.
            if cfile is not None and ok:
                got = {sym: store[sym] for sym in chunk if sym in store}
                cfile.write_text(json.dumps(got))
            print(f"  fetched {min(i + batch, len(tickers))}/{len(tickers)}", flush=True)
            time.sleep(pause)

        cols = {}
        for sym, bars in store.items():
            if not bars:
                continue
            d = pd.DataFrame(bars)
            d["t"] = pd.to_datetime(d["t"]).dt.tz_localize(None).dt.normalize()
            # Some symbols return more than one bar per calendar day -- a
            # paginated page boundary re-emitting a bar, or a corporate-action
            # split print. reindex() raises on duplicate labels rather than
            # silently picking one, so collapse to the last bar per day here.
            d = d.drop_duplicates(subset="t", keep="last")
            cols[sym] = d.set_index("t")[["o", "h", "l", "c", "v"]]
        if not cols:
            raise RuntimeError("Alpaca returned no bars")

        idx = pd.DatetimeIndex(sorted(set().union(*[c.index for c in cols.values()])))
        build = lambda k: pd.DataFrame({s: c[k].reindex(idx) for s, c in cols.items()},
                                       index=idx)
        o, h, l, c, v = (build(k) for k in ["o", "h", "l", "c", "v"])
        # adjustment=all means these bars are already split+dividend adjusted
        return Panel(o, h, l, c, v, c.copy(),
                     survivorship_free=True, source="alpaca")


def get_source(prefer: str = "auto"):
    """Pick the best available source, preferring survivorship-free ones."""
    if prefer in ("auto", "alpaca"):
        try:
            return AlpacaSource()
        except RuntimeError:
            if prefer == "alpaca":
                raise
            print("[data] Alpaca keys not set")
    print("[data] using Yahoo -- SURVIVOR-ONLY, returns will be biased HIGH")
    return YahooSource()
