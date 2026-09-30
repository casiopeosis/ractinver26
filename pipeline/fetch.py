"""
Pull everything the analysis needs from Yahoo Finance. Run it in YOUR terminal
(it needs normal internet access):

    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pipeline.fetch              # ~5-10 min, mostly the options step
    python -m pipeline.fetch --no-options # quick: prices, dividends, earnings dates only

Writes to data/:
    close.csv            daily closes, native currency (MXN for .MX, USD for SIC/ETF)
    dividends.csv        cash dividends per share by ex-date (the simulator does NOT pay them)
    earnings.csv         past + upcoming earnings timestamps per ticker
    options_events.csv   ATM straddle around the next earnings -> implied event move
    intraday_1m.csv      last 7 days of 1-minute bars for the SIC-lag check
    fetch_log.txt        every ticker that failed, and why

Every network call is wrapped: one bad ticker never kills the run.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import BENCH, FX, live

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONTEST_START, CONTEST_END = pd.Timestamp("2026-10-05"), pd.Timestamp("2026-11-13")
INTRADAY = ["NVDA", "UPST", "MARA", FX]

_log: list[str] = []


def log(msg: str) -> None:
    print(msg)
    _log.append(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}")


def retry(fn, *a, tries=3, wait=2.0, **kw):
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:  # yfinance raises all sorts; rate limits are common
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


# ---------------------------------------------------------------------------
def fetch_prices(yf, symbols: list[str]) -> None:
    log(f"prices: {len(symbols)} symbols, 2y daily")
    df = retry(yf.download, symbols, period="2y", interval="1d", auto_adjust=False,
               actions=True, progress=False, group_by="column", threads=True)
    close = df["Close"].dropna(how="all", axis=1)
    missing = sorted(set(symbols) - set(close.columns))
    for m in missing:
        log(f"  FAILED price {m} (check symbol in universe.py)")
    close.to_csv(DATA / "close.csv")
    divs = df["Dividends"].reindex(columns=close.columns)
    divs = divs.where(divs > 0).stack().dropna().rename("dividend").reset_index()
    divs.columns = ["date", "yahoo", "dividend"]
    divs.to_csv(DATA / "dividends.csv", index=False)
    log(f"  ok: {close.shape[1]} series, {len(close)} days; {len(divs)} dividend events")


def fetch_earnings(yf, symbols: list[str]) -> None:
    log(f"earnings dates: {len(symbols)} symbols")
    rows = []
    for s in symbols:
        try:
            ed = retry(yf.Ticker(s).get_earnings_dates, limit=12)
            if ed is None or ed.empty:
                continue
            for ts, r in ed.iterrows():
                rows.append(dict(yahoo=s, ts=ts, eps_est=r.get("EPS Estimate"),
                                 eps_actual=r.get("Reported EPS")))
        except Exception as e:
            log(f"  no earnings {s}: {type(e).__name__}")
        time.sleep(0.3)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "earnings.csv", index=False)
    log(f"  ok: {len(out)} earnings rows")


def reaction_day(ts: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    """US reports: after 16:00 ET -> next session reacts; before 09:30 -> same day."""
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    d = pd.Timestamp(t.date())
    if t.hour >= 16:
        return d + pd.offsets.BDay(1), "amc"
    if t.hour < 9 or (t.hour == 9 and t.minute < 30):
        return d, "bmo"
    return d + pd.offsets.BDay(1), "unknown"   # midnight stamps = time not announced


def _mid(row) -> float:
    b, a, last = row.get("bid", np.nan), row.get("ask", np.nan), row.get("lastPrice", np.nan)
    return (b + a) / 2 if (b > 0 and a > 0) else last


def fetch_options(yf, symbols: list[str]) -> None:
    """Implied earnings move from the ATM straddle of the first expiry after the
    reaction day. Straddle/spot ~ E|move| = sigma*sqrt(2/pi) -> sigma ~ 1.2533 x.
    The straddle also carries ordinary diffusion until expiry; analyze.py
    subtracts that using the pre-event ATM IV."""
    earn = pd.read_csv(DATA / "earnings.csv", parse_dates=["ts"]) if (DATA / "earnings.csv").exists() else None
    if earn is None or earn.empty:
        log("options: no earnings file, skipping"); return
    earn["ts"] = pd.to_datetime(earn["ts"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    upcoming = earn[(earn.ts > now) & (earn.ts < pd.Timestamp(CONTEST_END, tz="UTC") + pd.Timedelta(days=1))]
    upcoming = upcoming[upcoming.yahoo.isin(symbols)].sort_values("ts").drop_duplicates("yahoo")
    log(f"options: {len(upcoming)} tickers report before the contest ends")
    rows = []
    for _, e in upcoming.iterrows():
        s = e.yahoo
        try:
            tk = yf.Ticker(s)
            spot = float(retry(tk.history, period="5d")["Close"].iloc[-1])
            react, timing = reaction_day(e.ts)
            exps = [pd.Timestamp(x) for x in retry(lambda: tk.options)]
            post = [x for x in exps if x >= react]
            pre = [x for x in exps if x < react and x > pd.Timestamp(now.date())]
            if not post:
                log(f"  no post-event expiry {s}"); continue

            def atm(exp):
                ch = retry(tk.option_chain, exp.strftime("%Y-%m-%d"))
                c, p = ch.calls, ch.puts
                common = sorted(set(c.strike) & set(p.strike))
                if not common:
                    raise ValueError("no common strikes")
                k = min(common, key=lambda x: abs(x - spot))
                cr, pr = c[c.strike == k].iloc[0], p[p.strike == k].iloc[0]
                return (_mid(cr) + _mid(pr)) / spot, float(np.nanmean([cr.impliedVolatility, pr.impliedVolatility]))

            straddle_pct, iv_post = atm(post[0])
            iv_pre = atm(pre[-1])[1] if pre else np.nan
            rows.append(dict(yahoo=s, earnings_ts=e.ts, reaction_day=react.date(), timing=timing,
                             spot=spot, expiry=post[0].date(),
                             days_to_expiry=int(np.busday_count(now.date(), post[0].date())),
                             straddle_pct=straddle_pct, iv_post=iv_post, iv_pre=iv_pre))
        except Exception as ex:
            log(f"  options failed {s}: {type(ex).__name__}: {ex}")
        time.sleep(0.5)
    pd.DataFrame(rows).to_csv(DATA / "options_events.csv", index=False)
    log(f"  ok: {len(rows)} implied moves")


def fetch_intraday(yf) -> None:
    log("intraday: 1m bars, last 7 days")
    df = retry(yf.download, INTRADAY, period="7d", interval="1m", progress=False,
               group_by="column", auto_adjust=False)
    df["Close"].to_csv(DATA / "intraday_1m.csv")
    log(f"  ok: {len(df)} bars")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-options", action="store_true")
    ap.add_argument("--only", choices=["prices", "earnings", "options", "intraday"])
    args = ap.parse_args()
    import yfinance as yf

    DATA.mkdir(exist_ok=True)
    uni = live()
    all_syms = [r["yahoo"] for r in uni] + [FX] + list(BENCH.values())
    us_syms = [r["yahoo"] for r in uni if r["kind"] == "sic"]
    mx_syms = [r["yahoo"] for r in uni if r["kind"] == "mx"]
    steps = {
        "prices": lambda: fetch_prices(yf, all_syms),
        "earnings": lambda: fetch_earnings(yf, us_syms + mx_syms),
        "options": lambda: fetch_options(yf, us_syms),
        "intraday": lambda: fetch_intraday(yf),
    }
    todo = [args.only] if args.only else [k for k in steps if not (k == "options" and args.no_options)]
    t0 = time.time()
    for k in todo:
        try:
            steps[k]()
        except Exception as e:
            log(f"STEP {k} FAILED: {type(e).__name__}: {e}")
    (DATA / "fetch_log.txt").write_text("\n".join(_log) + "\n")
    print(f"\ndone in {time.time() - t0:.0f}s -> {DATA}")


if __name__ == "__main__":
    main()
