"""
Turn data/ into the four tables that drive decisions:

    python -m pipeline.analyze      -> results/report.md + CSVs

1. vol_table.csv        every live instrument's volatility IN PESOS (SIC = USD x FX),
                        ex-earnings base vol, correlation to the IPC, past earnings moves
2. pairs.csv            best two-name portfolios at the 50% purchase cap: the realistic
                        variance ceiling, measured instead of assumed
3. event_calendar.csv   earnings inside Oct 5 - Nov 13 with option-implied jump size
4. dividend_warnings.csv  projected ex-dates in the window (the simulator doesn't pay
                        dividends, so holding through one is a pure loss)
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import CONTEST_END, CONTEST_START, DATA, reaction_day
from .universe import BENCH, FX, live

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
ANN = np.sqrt(252)
W_CAP = 0.495            # size to 49.5%: market orders fill at the NEXT print
SQRT_2_OVER_PI_INV = 1.2533


# ---------------------------------------------------------------------------
def load_mxn_returns() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    close = pd.read_csv(DATA / "close.csv", index_col=0, parse_dates=True)
    close.index = pd.to_datetime(close.index, utc=True).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    fx = close[FX].ffill()
    uni = [r for r in live() if r["yahoo"] in close.columns]
    px = {}
    for r in uni:
        s = close[r["yahoo"]]
        px[r["name"]] = (s * fx) if r["usd"] else s
    px = pd.DataFrame(px)
    for k, sym in BENCH.items():
        if sym in close.columns:
            px[f"^{k}"] = close[sym] * (fx if k == "SPX" else 1.0)
    rets = np.log(px).diff()
    return px, rets, uni


def earnings_reaction_days() -> dict[str, list[pd.Timestamp]]:
    f = DATA / "earnings.csv"
    if not f.exists():
        return {}
    e = pd.read_csv(f)
    if e.empty:
        return {}
    e["ts"] = pd.to_datetime(e["ts"], utc=True)
    out: dict[str, list] = {}
    for _, r in e.iterrows():
        out.setdefault(r.yahoo, []).append(reaction_day(r.ts)[0])
    return out


def winsorize(x: pd.Series | pd.DataFrame, k: float = 5.0):
    """Clip returns at +-k robust sigmas (MAD). One 100% gap (MRNA, Aug-2026) should
    not define a 30-day variance budget: jumps get modelled as events, not as vol."""
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return x.clip(med - k * mad, med + k * mad, axis=1 if isinstance(x, pd.DataFrame) else None)


def vol_table(px, rets, uni, react) -> pd.DataFrame:
    ipc = rets.get("^IPC")
    rows = []
    for r in uni:
        n = r["name"]
        x = rets[n].dropna()
        if len(x) < 60:
            continue
        rd = set(pd.Timestamp(d).normalize() for d in react.get(r["yahoo"], []))
        is_event = x.index.isin(list(rd))
        x250 = x.iloc[-250:]
        base = x250[~is_event[-len(x250):]]
        past_moves = x[is_event].abs()
        corr = x250.corr(ipc.reindex(x250.index)) if ipc is not None else np.nan
        rows.append(dict(
            name=n, kind=r["kind"], price_mxn=px[n].dropna().iloc[-1],
            vol_60d=x.iloc[-60:].std() * ANN, vol_250d=x250.std() * ANN,
            vol_ex_earnings=base.std() * ANN, vol_robust_60d=winsorize(x.iloc[-60:]).std() * ANN,
            max_abs_day_60d=x.iloc[-60:].abs().max(), corr_ipc=corr,
            n_past_earnings=len(past_moves),
            mean_abs_earnings_move=past_moves.mean() if len(past_moves) else np.nan,
            max_abs_earnings_move=past_moves.max() if len(past_moves) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values("vol_robust_60d", ascending=False)


def best_pairs(rets, vt: pd.DataFrame, top: int = 60, keep: int = 25) -> pd.DataFrame:
    """sigma of W_CAP/W_CAP (+ remainder cash) using 60d covariance.
    Under a cap, correlation is your friend: sigma_p = w*sigma*sqrt(2(1+rho))."""
    names = vt.head(top).name.tolist()
    R = winsorize(rets[names].iloc[-60:].dropna(axis=1, thresh=50).fillna(0.0))
    C = R.cov().values * 252
    idx = {n: i for i, n in enumerate(R.columns)}
    ipc = rets.get("^IPC")
    rows = []
    for a, b in combinations(R.columns, 2):
        i, j = idx[a], idx[b]
        var = W_CAP ** 2 * (C[i, i] + C[j, j] + 2 * C[i, j])
        rho = C[i, j] / np.sqrt(C[i, i] * C[j, j])
        port = W_CAP * (R[a] + R[b])
        rows.append(dict(a=a, b=b, vol_pair=np.sqrt(var), rho=rho,
                         corr_ipc=port.corr(ipc.reindex(port.index)) if ipc is not None else np.nan))
    return pd.DataFrame(rows).sort_values("vol_pair", ascending=False).head(keep)


def event_calendar(vt: pd.DataFrame, uni) -> pd.DataFrame:
    f = DATA / "earnings.csv"
    if not f.exists():
        return pd.DataFrame()
    e = pd.read_csv(f)
    e["ts"] = pd.to_datetime(e["ts"], utc=True)
    y2n = {r["yahoo"]: r["name"] for r in uni}
    rows = []
    for _, r in e.iterrows():
        d, timing = reaction_day(r.ts)
        if CONTEST_START <= d <= CONTEST_END and r.yahoo in y2n:
            rows.append(dict(name=y2n[r.yahoo], yahoo=r.yahoo, report_ts=r.ts, timing=timing, reaction_day=d.date()))
    cal = pd.DataFrame(rows).drop_duplicates("name")
    if cal.empty:
        return cal
    cal = cal.merge(vt[["name", "vol_ex_earnings", "mean_abs_earnings_move", "max_abs_earnings_move"]],
                    on="name", how="left")
    of = DATA / "options_events.csv"
    if of.exists() and of.stat().st_size > 5:
        o = pd.read_csv(of)
        if not o.empty:
            cal = cal.merge(o[["yahoo", "straddle_pct", "iv_post", "iv_pre", "days_to_expiry"]], on="yahoo", how="left")
            # term structure: the post-event expiry's IV carries diffusion + jump,
            # the pre-event expiry's IV only diffusion -> jump var = (iv_post^2 - iv_pre^2) * tau
            tau = cal.days_to_expiry.clip(lower=1) / 252
            ev = (cal.iv_post ** 2 - cal.iv_pre ** 2) * tau
            cal["implied_event_sigma"] = np.sqrt(ev.clip(lower=0))
    # historical fallback: E|X| = sigma sqrt(2/pi)
    cal["hist_event_sigma"] = SQRT_2_OVER_PI_INV * cal.mean_abs_earnings_move
    imp = cal["implied_event_sigma"] if "implied_event_sigma" in cal else pd.Series(np.nan, index=cal.index)
    # implied is the market's forecast but noisy weeks out; fall back to history when it degenerates
    cal["event_sigma"] = imp.where(imp > 0.01, cal.hist_event_sigma)
    return cal.sort_values("reaction_day")


def dividend_warnings(uni) -> pd.DataFrame:
    """Project the next ex-date from each payer's median gap; flag those in the window.
    yield_per_event = what you lose by holding through that ex-date."""
    f = DATA / "dividends.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f).dropna(subset=["dividend"])
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_localize(None).dt.normalize()
    close = pd.read_csv(DATA / "close.csv", index_col=0)
    y2n = {r["yahoo"]: r["name"] for r in uni}
    rows = []
    for sym, g in d.groupby("yahoo"):
        if sym not in y2n or len(g) < 2 or sym not in close:
            continue
        g = g.sort_values("date")
        gap = g.date.diff().median()
        if gap.days < 20:          # noise / special dividends
            continue
        nxt = g.date.iloc[-1]
        while nxt < CONTEST_START:
            nxt += gap
        if nxt <= CONTEST_END:
            last_px = close[sym].dropna().iloc[-1]
            rows.append(dict(name=y2n[sym], projected_ex_date=nxt.date(), every_days=gap.days,
                             yield_per_event=g.dividend.iloc[-1] / last_px))
    out = pd.DataFrame(rows)
    return out.sort_values("projected_ex_date") if not out.empty else out


def _md(df: pd.DataFrame, cols, fmt: dict, n: int) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    d = df[cols].head(n).copy()
    for c, f in fmt.items():
        if c in d:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f.format(v))
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.values) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    px, rets, uni = load_mxn_returns()
    react = earnings_reaction_days()
    vt = vol_table(px, rets, uni, react)
    pairs = best_pairs(rets, vt)
    cal = event_calendar(vt, uni)
    divs = dividend_warnings(uni)
    for name, df in (("vol_table", vt), ("pairs", pairs), ("event_calendar", cal), ("dividend_warnings", divs)):
        df.to_csv(OUT / f"{name}.csv", index=False)

    pct = "{:.0%}"
    rep = [f"# Universe report — data through {px.index[-1].date()}\n",
           "All vols annualised, in **pesos** (SIC/ETF = USD price × USDMXN).\n",
           "## Highest-vol instruments (60d)\n",
           "Sorted by robust 60d vol (returns clipped at ±5 MAD-sigmas); `max_day` = biggest 1-day move in 60d.\n",
           _md(vt, ["name", "kind", "vol_robust_60d", "vol_60d", "vol_250d", "max_abs_day_60d", "corr_ipc",
                    "mean_abs_earnings_move"],
               {"vol_robust_60d": pct, "vol_60d": pct, "vol_250d": pct, "max_abs_day_60d": "{:.0%}",
                "corr_ipc": "{:.2f}", "mean_abs_earnings_move": "{:.1%}"}, 30),
           f"\n## Variance ceiling: best pairs at {W_CAP:.1%} / {W_CAP:.1%}\n",
           _md(pairs, ["a", "b", "vol_pair", "rho", "corr_ipc"],
               {"vol_pair": pct, "rho": "{:.2f}", "corr_ipc": "{:.2f}"}, 15),
           f"\n## High-variance events inside the contest ({CONTEST_START.date()} – {CONTEST_END.date()})\n",
           f"event σ ≥ 8% or base vol ≥ 50%. All {len(cal)} events are in event_calendar.csv. "
           "`unknown` timing = hold across both the report day and the next.\n",
           _md(cal[(cal.event_sigma >= 0.08) | (cal.vol_ex_earnings >= 0.5)] if not cal.empty else cal,
               [c for c in ["reaction_day", "name", "timing", "event_sigma", "implied_event_sigma",
                            "hist_event_sigma", "max_abs_earnings_move", "vol_ex_earnings"] if c in cal],
               {"event_sigma": "{:.1%}", "implied_event_sigma": "{:.1%}", "hist_event_sigma": "{:.1%}",
                "max_abs_earnings_move": "{:.1%}", "vol_ex_earnings": pct}, 80),
           "\n## Dividend traps (projected ex-dates in the window)\n",
           _md(divs, ["projected_ex_date", "name", "every_days", "yield_per_event"], {"yield_per_event": "{:.2%}"}, 50)]
    (OUT / "report.md").write_text("\n".join(rep))
    print(f"wrote {OUT / 'report.md'}")
    if not pairs.empty:
        top = pairs.iloc[0]
        print(f"variance ceiling ~ {top.vol_pair:.0%} ({top.a} + {top.b}, rho {top.rho:.2f})")


if __name__ == "__main__":
    main()
