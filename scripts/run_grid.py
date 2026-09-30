"""
Scan your strategy space and find the risk level that maximises P(1st place).

    python scripts/run_grid.py                 # full run
    python scripts/run_grid.py --sims 5000     # quick run

Outputs in results/:
    calibration.json   fitted field-wildness multiplier (cached; --recalibrate to redo)
    grid.csv           every (field, N, strategy) -> P(win), P(top3/5/10), median, P(loss)
    p_win.png          the picture
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament import Contest, Exposure, default_field, draw_scenarios, rank_probabilities  # noqa: E402
from tournament.calibrate import calibrate_lambda, scale_field  # noqa: E402

RESULTS = ROOT / "results"

# Rough ceiling on idiosyncratic vol reachable WITHOUT leverage: concentrated in
# one or two volatile BMV names. An assumption -- check it against real data.
FEASIBLE_IDIO_MAX = 0.45

IDIO_GRID = np.round(np.linspace(0.0, 2.5, 26), 3)
B_H_GRID = (0.0, 1.0)       # independent of the crowd vs. riding the crowd's names
B_M_GRID = (1.0, 1.5)       # plain vs. leveraged market exposure (if the simulator allows)
N_GRID = (300, 500, 1000)


def get_lambda(contest, recalibrate: bool) -> float:
    path = RESULTS / "calibration.json"
    if path.exists() and not recalibrate:
        return json.loads(path.read_text())["lambda"]
    print("calibrating field to 2025 national winner (+70.6%, N~40k)...")
    lam = calibrate_lambda(default_field(), contest, draw_scenarios(contest, 3000, seed=123))
    path.write_text(json.dumps({"lambda": lam, "target_winner": 0.706, "n_ref": 40_000}, indent=2))
    return lam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw p_win.png from grid.csv")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.plot_only:
        import csv
        rows = [{k: (v if k == "field" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(open(RESULTS / "grid.csv"))]
        for r in rows:
            r["n"] = int(r["n"])
        plot(rows, json.loads((RESULTS / "calibration.json").read_text())["lambda"])
        return

    contest = Contest()
    lam = get_lambda(contest, args.recalibrate)
    fields = {
        "calibrated": scale_field(default_field(), lam),   # tame, matches 2025
        "wild": default_field(),                           # raw priors: aggressive cohort
    }
    sc = draw_scenarios(contest, args.sims, seed=2026)     # common random numbers

    rows = []
    for (fname, base), n, b_M, b_H, iv in product(fields.items(), N_GRID, B_M_GRID, B_H_GRID, IDIO_GRID):
        r = rank_probabilities(Exposure(b_M=b_M, b_H=b_H, idio_vol=float(iv)), base.with_n(n), contest, sc)
        rows.append(dict(field=fname, n=n, b_M=b_M, b_H=b_H, idio_vol=float(iv),
                         p_win=r.p_win, p_win_se=r.p_win_se, p_top3=r.p_top[2],
                         p_top5=r.p_top[4], p_top10=r.p_top[9],
                         median_ret=r.median_return, p_loss=r.p_loss))

    import csv
    with open(RESULTS / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    print(f"\nfield wildness lambda = {lam:.3f}   (1.0 = raw priors)\n")
    print(f"{'field':<11}{'N':>5}{'b_M':>5}{'b_H':>5} | {'best idio':>9} {'P(win)':>8} {'x naive':>8}"
          f" {'P(top5)':>8} {'median':>8} {'P(loss)':>8}")
    for fname, n, b_M, b_H in product(fields, N_GRID, B_M_GRID, B_H_GRID):
        sub = [r for r in rows if (r["field"], r["n"], r["b_M"], r["b_H"]) == (fname, n, b_M, b_H)]
        best = max(sub, key=lambda r: r["p_win"])
        print(f"{fname:<11}{n:>5}{b_M:>5}{b_H:>5} | {best['idio_vol']:>9.0%} {best['p_win']:>8.2%}"
              f" {best['p_win'] * (n + 1):>7.1f}x {best['p_top5']:>8.1%} {best['median_ret']:>+8.1%}"
              f" {best['p_loss']:>8.0%}")
    print("\n'x naive' = P(win) relative to 1/(N+1), a random seat at the table.")

    plot(rows, lam)


def plot(rows, lam) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "monospace", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    shades = {300: "#9a9a9a", 500: "#4a4a4a", 1000: "#000000"}
    for ax, fname in zip(axes, ("calibrated", "wild")):
        for n, b_H in product(N_GRID, B_H_GRID):
            sub = [r for r in rows if r["field"] == fname and r["n"] == n and r["b_M"] == 1.0 and r["b_H"] == b_H]
            x = [r["idio_vol"] for r in sub]
            y = [r["p_win"] * (n + 1) for r in sub]
            ax.plot(x, y, color=shades[n], ls="-" if b_H == 0 else "--", lw=1.4,
                    label=f"N={n}, {'own picks' if b_H == 0 else 'crowd names'}")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 400)
        ax.axhline(1, color="#c0392b", lw=0.8)
        ax.axvspan(FEASIBLE_IDIO_MAX, IDIO_GRID[-1], color="#000", alpha=0.05, lw=0)
        ax.text(FEASIBLE_IDIO_MAX + 0.03, 0.13, "needs leverage /\nvery volatile names",
                va="bottom", fontsize=7, color="#555")
        ax.set_title(f"{fname} field" + (f"  (λ={lam:.2f})" if fname == "calibrated" else "  (raw priors)"),
                     loc="left", fontweight="bold")
        ax.set_xlabel("your idiosyncratic vol (annualised)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("P(1st) ÷ 1/(N+1)   [edge vs. random seat]")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Reto Actinver 2026 — how much risk maximises P(1st place)?  (β_M = 1, 30 trading days)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "p_win.png", dpi=160)
    print(f"saved {RESULTS / 'p_win.png'}")


if __name__ == "__main__":
    main()
