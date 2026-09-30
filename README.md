# ractinver26 — tournament model for Reto Actinver 2026

The contest pays for **rank**, not returns. This repo answers the first question
of any tournament strategy: *how much risk, and what kind, maximises P(1st)?*
Signals come later; this tells you what shape of bet they need to produce.

```
source .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q                     # engine vs brute force, pipeline on fake data
python -m pipeline.fetch                # YOUR terminal (needs internet): ~5-10 min -> data/
python -m pipeline.analyze              # -> results/report.md (+ CSVs)
python -m pipeline.sic_lag              # simulator SIC quotes vs New York x FX
python scripts/run_grid.py              # tournament risk scan -> results/p_win.png
```

## Confirmed rules (Actinver, by email + platform tests, Sep 28)

- **50% cap is checked only at purchase**, against your whole position in that name
  (buying 40% and then another 15% is blocked). A winner can drift past 50%.
- **5 instruments**: any mix of stocks, ETFs, FIBRAs or funds, *traded at some point*.
  You don't have to hold them.
- **Commission 0.10% + IVA = 0.116% per side**, verified to the cent.
- **No dividends credited**, so holding through an ex-date is a pure loss. See the dividend traps in the report.
- **Automation is prohibited** for placing orders. The pipeline only analyses; every
  order is entered by hand. Nothing here touches retoactinver.com.
- **Multiple accounts are allowed** (one per institutional email for Universitario; personal
  email for General). Only the single largest prize is paid.
- Dead tickers (CPE, MRO, PARA) can still be bought. Don't.

## Layout

- `tournament/`: rank-probability engine, calibration, prize objective
- `pipeline/universe.py`: the simulator's instruments mapped to Yahoo symbols
- `pipeline/fetch.py`: data download (run locally)
- `pipeline/analyze.py`: peso vols, variance ceiling (best pairs), event calendar, dividend traps
- `pipeline/sic_lag.py`: checks your logged simulator prices (`state/sic_log.csv`) against NY

## Model

Every portfolio over the 30 trading days (Oct 5 – Nov 13):

    log R = m + b_M·M + b_H·H + s·ε

`M` = market (IPC), `H` = the crowd's "hot basket", `ε` = your own idiosyncratic
shock (unit-variance Student-t, df=4, fat tails). `m` includes the Itô drag
`−½σ²T`, so volatility costs you median return.

The field is a mix of archetypes (`tournament/model.py`): dormant 35%, indexer
20%, hot-chaser 25%, active trader 15%, gambler 5%.

**Engine (`tournament/engine.py`).** Conditional on (M, H), rivals are
independent, so

    P(win | M,H,you) = Π_a F_a(x)^{n_a}      #ahead ~ Σ_a Binomial(n_a, 1−F_a)

in closed form. We only simulate the common factors and your shock, so the cost
doesn't depend on N and the variance is far lower than brute force
(Rao–Blackwellisation). `tournament/bruteforce.py` simulates every rival and
serves as the check in `tests/`.

**Calibration (`tournament/calibrate.py`).** One knob, λ, scales the whole
field's idiosyncratic risk. It is fit so that the *median* simulated national
winner (N = 40k) equals the real 2025 winner, +70.6%. Result: **λ ≈ 0.24**, so
real participants are about 4× tamer than the raw priors. The "wild" field
(λ = 1) is kept as a stress case for an aggressive cohort like a finance-heavy
university.

## What it says (first run)

1. **Only risk you don't share with the field counts.** Market leverage
   (`b_M` 1 → 1.5) and riding the crowd's names (`b_H`) barely move P(win).
   Common factors lift or sink everyone together, so they cancel out of the
   ranking. The lever is idiosyncratic vol.

2. **There's a closed-form optimum.** If you need to beat a winning log return
   `c`, then P(log R > c) with log R ~ N(−½σ²T, σ²T) is maximised at

       σ* √T = √(2c)    ⇒    σ* = √(2c / T)

   For N = 500 (calibrated), the winning score is about +18%, so c ≈ 0.165 and
   σ* ≈ 166% annualised. The grid finds 160%. The theory checks out.

3. **That optimum isn't reachable, so the constraint binds: take the most
   idiosyncratic risk the platform allows.** Without leverage, about 40–50%
   annualised idio vol (one or two volatile BMV names) is a realistic ceiling.
   Results for N = 500:

   | idio vol | P(win) calibrated | edge vs random | P(win) wild | median return |
   |---|---|---|---|---|
   | 20% | 3.8% | 19× | 0.03% | +0.5% |
   | 30% | 7.1% | 35× | 0.12% | +0.2% |
   | 40% | 10.3% | 52× | 0.29% | −0.2% |
   | 50% | 13.1% | 66× | 0.55% | −0.8% |

   The median barely suffers at these levels, so the variance is nearly free.

4. **How wild the field is matters far more than N.** At 40% vol, P(win) is
   10.3% against the calibrated field and 0.3% against the wild one, a 35×
   gap. Going from N = 300 to 1000 changes things much less. Measuring the
   field (practice-week leaderboard, the first weeks of the contest) is the
   highest-value information you can get.

## Ways to buy idiosyncratic variance without leverage (to verify)

- **Concentration.** One or two high-vol names. First check how the rule "trade
  or hold ≥ 5 different stocks" is enforced: at any point in time, or at some
  point during the contest.
- **Event rotation.** Hold whichever stock reports earnings that day. Q3
  season falls inside the window. About 15 events × a ~5% earnings-day move
  gives a horizon σ ≈ 19%, which is roughly 55% annualised, added on top of
  normal volatility.
- **Leveraged or inverse ETFs, if the simulator's universe includes any.**

## Next steps

- [ ] Replace parametric shocks with bootstrapped **real BMV returns** for the
      simulator's actual universe, which gives the feasible vol ceiling from data.
- [ ] Recalibrate λ from the **practice-week leaderboard** (Sep 28 – Oct 2).
- [ ] **Dynamic version**: re-optimise risk each week given your current rank
      and the leaders' returns. The optimal σ changes as the gap and the time
      left shrink.
- [ ] Weekly prizes as separate mini-tournaments.
