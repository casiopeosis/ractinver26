"""
Does the simulator's SIC price track New York in real time?

    python -m pipeline.sic_lag

Compares each reading in state/sic_log.csv (what you saw on the platform,
Mexico City time) with the NY 1-minute price x USDMXN at that minute and at
lags of 0-30 min. If the best match is at lag > 0, SIC quotes (and market-order
fills) trail New York. Needs data/intraday_1m.csv from fetch.py, which only
covers the last 7 days: run fetch soon after logging.
"""
from __future__ import annotations

import pandas as pd

from .fetch import DATA, FX
from .universe import SIC

ROOT = DATA.parent
LAGS = [0, 1, 2, 5, 10, 15, 30]


def main() -> None:
    log = pd.read_csv(ROOT / "state" / "sic_log.csv")
    # Mexico City has been UTC-6 all year since 2022 (no DST)
    log["ts_utc"] = pd.to_datetime(log.timestamp_cdmx) + pd.Timedelta(hours=6)
    bars = pd.read_csv(DATA / "intraday_1m.csv", index_col=0)
    bars.index = pd.to_datetime(bars.index, utc=True).tz_localize(None)
    bars = bars.sort_index().ffill()
    rows = []
    for _, r in log.iterrows():
        sym = SIC.get(r["name"], r["name"])
        if sym not in bars:
            continue
        row = dict(time=r.timestamp_cdmx, name=r["name"], sim=r.sim_price_mxn)
        for lag in LAGS:
            t = r.ts_utc - pd.Timedelta(minutes=lag)
            b = bars.loc[:t].iloc[-1] if (bars.index <= t).any() else None
            if b is None:
                continue
            ny_mxn = b[sym] * b[FX]
            row[f"dev_lag{lag}"] = r.sim_price_mxn / ny_mxn - 1
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        print("no overlap between your log and the 1m bars (older than 7 days?)"); return
    pd.set_option("display.float_format", "{:+.2%}".format)
    print(out.to_string(index=False))
    devs = out.filter(like="dev_lag").abs().mean()
    print("\nmean |deviation| by assumed lag (minutes):")
    print(devs.to_string())
    print(f"\nbest-matching lag: {devs.idxmin().replace('dev_lag', '')} min")


if __name__ == "__main__":
    main()
