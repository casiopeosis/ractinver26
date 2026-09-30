"""End-to-end run of analyze.py + sic_lag.py on fake data shaped like yfinance's."""
import numpy as np
import pandas as pd
import pytest

from pipeline import analyze, fetch, sic_lag
from pipeline.universe import BENCH, FX, live


@pytest.fixture
def fake(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "results"
    data.mkdir(); (tmp_path / "state").mkdir()
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-10-01", "2026-09-28")
    syms = [r["yahoo"] for r in live()] + [FX] + list(BENCH.values())
    vol = {s: 0.25 for s in syms}
    vol.update({"MARA": 1.0, "RIOT": 1.0, "UPST": 0.9, FX: 0.1})
    close = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0, vol[s] / np.sqrt(252), len(days))))
                          for s in syms}, index=days)
    common = rng.normal(0, 0.8 / np.sqrt(252), len(days))          # make MARA/RIOT correlated
    close["MARA"] = 20 * np.exp(np.cumsum(common + rng.normal(0, 0.4 / np.sqrt(252), len(days))))
    close["RIOT"] = 12 * np.exp(np.cumsum(common + rng.normal(0, 0.4 / np.sqrt(252), len(days))))
    close[FX] = 18.5
    close.to_csv(data / "close.csv")
    pd.DataFrame({"date": pd.date_range("2025-01-28", "2026-09-28", freq="28D"),
                  "yahoo": "AGNC", "dividend": 0.12}).to_csv(data / "dividends.csv", index=False)
    pd.DataFrame({"yahoo": ["UPST", "NVDA", "UPST"],
                  "ts": ["2026-11-03 16:05:00-05:00", "2026-11-18 16:20:00-05:00", "2026-08-05 16:05:00-04:00"],
                  "eps_est": [0.1, 1.0, 0.1], "eps_actual": [np.nan, np.nan, 0.2]}).to_csv(data / "earnings.csv", index=False)
    pd.DataFrame({"yahoo": ["UPST"], "straddle_pct": [0.20], "iv_post": [1.0], "iv_pre": [0.8], "days_to_expiry": [15]}
                 ).to_csv(data / "options_events.csv", index=False)
    t = pd.date_range("2026-09-28 13:00", "2026-09-28 20:00", freq="1min", tz="UTC")
    pd.DataFrame({"NVDA": 222.0, "UPST": 22.2, "MARA": 11.8, FX: 18.55}, index=t).to_csv(data / "intraday_1m.csv")
    pd.DataFrame({"timestamp_cdmx": ["2026-09-28 10:20"], "name": ["NVDA"], "sim_price_mxn": [4118.10]}
                 ).to_csv(tmp_path / "state" / "sic_log.csv", index=False)
    for m in (fetch, analyze, sic_lag):
        monkeypatch.setattr(m, "DATA", data, raising=False)
    monkeypatch.setattr(analyze, "OUT", out)
    monkeypatch.setattr(sic_lag, "ROOT", tmp_path)
    return tmp_path


def test_analyze_end_to_end(fake):
    analyze.main()
    res = fake / "results"
    pairs = pd.read_csv(res / "pairs.csv")
    assert {pairs.iloc[0].a, pairs.iloc[0].b} == {"MARA", "RIOT"}       # correlated high-vol pair wins
    cal = pd.read_csv(res / "event_calendar.csv")
    assert cal.name.tolist() == ["UPST"]                                   # NVDA reports after Nov 13
    assert str(cal.reaction_day.iloc[0]) == "2026-11-04"                   # after-close -> next day
    assert abs(cal.implied_event_sigma.iloc[0] - np.sqrt(0.36 * 15 / 252)) < 1e-9
    divs = pd.read_csv(res / "dividend_warnings.csv")
    assert "AGNC" in divs.name.tolist()
    assert "# Universe report" in (res / "report.md").read_text()


def test_sic_lag(fake, capsys):
    sic_lag.main()
    out = capsys.readouterr().out
    assert "best-matching lag" in out and "+0.00%" in out                  # 222 * 18.55 = 4118.10
