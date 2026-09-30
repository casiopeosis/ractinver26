"""
Rank-probability engine: conditional Monte Carlo.

The naive way to get P(you finish 1st) is to simulate all N rivals S times:
O(N * S) draws, and p_win ~ 1/N is a rare event, so the estimate is noisy.

The trick: rivals only interact through the common factors (M, H). Condition on
them and each rival of archetype a independently beats you with probability

    q_a(s) = 1 - F_a( x_s | M_s, H_s )

where x_s is your log return in scenario s and F_a is that archetype's
conditional CDF (a scaled Student-t, known in closed form). Then

    P(win | s)            = prod_a F_a^{n_a}                  (exact)
    #rivals ahead | s     ~ sum_a Binomial(n_a, q_a)          (exact)

and we only Monte-Carlo over (M, H, your own shock): O(S * #archetypes).
Cost no longer grows with N, and the variance is far lower (Rao-Blackwellization:
we replaced an indicator with its conditional expectation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .model import Contest, Exposure, Field


# ---------- unit-variance Student-t helpers ----------------------------------
def _t_scale(df: float) -> float:
    """Std of a standard t_df; divide by it to get unit variance."""
    return 1.0 if np.isinf(df) else float(np.sqrt(df / (df - 2.0)))


def unit_t_sample(rng: np.random.Generator, df: float, size) -> np.ndarray:
    if np.isinf(df):
        return rng.standard_normal(size)
    return rng.standard_t(df, size) / _t_scale(df)


def unit_t_logcdf(z: np.ndarray, df: float) -> np.ndarray:
    if np.isinf(df):
        return stats.norm.logcdf(z)
    return stats.t.logcdf(z * _t_scale(df), df)


def unit_t_logsf(z: np.ndarray, df: float) -> np.ndarray:
    if np.isinf(df):
        return stats.norm.logsf(z)
    return stats.t.logsf(z * _t_scale(df), df)


# ---------- scenario generation ----------------------------------------------
@dataclass
class Scenarios:
    M: np.ndarray       # horizon market shock (log-return units)
    H: np.ndarray       # horizon hot-basket shock
    eps_me: np.ndarray  # your unit idiosyncratic shock


def draw_scenarios(contest: Contest, n_sims: int, seed: int | None = 0,
                   me_df: float = 4.0) -> Scenarios:
    """Common random numbers: reuse the same Scenarios across strategies so
    comparisons between strategies aren't polluted by sampling noise."""
    rng = np.random.default_rng(seed)
    f, sT = contest.factors, np.sqrt(contest.T)
    return Scenarios(
        M=f.vol_M * sT * unit_t_sample(rng, f.df, n_sims),
        H=f.vol_H * sT * unit_t_sample(rng, f.df, n_sims),
        eps_me=unit_t_sample(rng, me_df, n_sims),
    )


def log_return(e: Exposure, contest: Contest, sc: Scenarios, eps: np.ndarray) -> np.ndarray:
    sT = np.sqrt(contest.T)
    return (e.log_mean(contest.factors, contest.T)
            + e.b_M * sc.M + e.b_H * sc.H + e.idio_vol * sT * eps)


def _cond_logF_logQ(x: np.ndarray, e: Exposure, contest: Contest, sc: Scenarios):
    """log P(rival < x | factors) and log P(rival > x | factors)."""
    centre = e.log_mean(contest.factors, contest.T) + e.b_M * sc.M + e.b_H * sc.H
    s = e.idio_vol * np.sqrt(contest.T)
    if s == 0.0:  # degenerate: rival's return is known given the factors
        above = x > centre
        return (np.where(above, 0.0, -np.inf), np.where(above, -np.inf, 0.0))
    z = (x - centre) / s
    return unit_t_logcdf(z, e.df), unit_t_logsf(z, e.df)


# ---------- main API ---------------------------------------------------------
@dataclass
class RankResult:
    p_win: float
    p_win_se: float
    p_top: np.ndarray          # p_top[k-1] = P(rank <= k), k = 1..k_max
    median_return: float       # your median simple return over the contest
    p_loss: float              # P(your return < 0)

    def summary(self) -> str:
        tops = ", ".join(f"top{k}={p:.3%}" for k, p in zip((1, 3, 5, 10), self.p_top[[0, 2, 4, 9]])
                         if k <= len(self.p_top))
        return (f"P(win)={self.p_win:.3%} ±{1.96 * self.p_win_se:.3%} | {tops} | "
                f"median ret={self.median_return:+.1%} | P(loss)={self.p_loss:.1%}")


def rank_probabilities(me: Exposure, field: Field, contest: Contest,
                       sc: Scenarios, k_max: int = 10) -> RankResult:
    x = log_return(me, contest, sc, sc.eps_me)
    counts = field.counts()
    S = x.shape[0]

    # P(win | s) in log space: sum_a n_a * log F_a
    log_pwin = np.zeros(S)
    # distribution of #rivals ahead, truncated at k_max - 1 (enough for top-k_max)
    pmf = np.zeros((S, k_max)); pmf[:, 0] = 1.0
    j = np.arange(k_max)

    for arch, n_a in zip(field.archetypes, counts):
        if n_a == 0:
            continue
        logF, logQ = _cond_logF_logQ(x, arch.exposure, contest, sc)
        log_pwin += n_a * logF
        q = np.exp(logQ)
        b = stats.binom.pmf(j[None, :], n_a, q[:, None])            # (S, k_max)
        new = np.zeros_like(pmf)
        for jj in range(k_max):                                      # truncated convolution
            new[:, jj] = np.sum(pmf[:, : jj + 1] * b[:, jj::-1], axis=1)
        pmf = new

    pwin_s = np.exp(log_pwin)
    p_top = np.cumsum(pmf, axis=1).mean(axis=0)
    R = np.expm1(x)
    return RankResult(
        p_win=float(pwin_s.mean()),
        p_win_se=float(pwin_s.std(ddof=1) / np.sqrt(S)),
        p_top=p_top,
        median_return=float(np.median(R)),
        p_loss=float((R < 0).mean()),
    )


def field_max_quantiles(field: Field, contest: Contest, sc: Scenarios,
                        qs=(0.1, 0.5, 0.9), grid=None) -> dict[float, float]:
    """Quantiles of the best rival's simple return: 'what score wins?'.

    P(max <= y) = E_s[ prod_a F_a(y | s)^{n_a} ], evaluated on a grid of y.
    Useful for calibration against real leaderboards.
    """
    if grid is None:
        grid = np.log1p(np.linspace(-0.3, 3.0, 661))
    counts = field.counts()
    grid = np.asarray(grid)
    # broadcast: rows = grid points, cols = scenarios
    sc2 = Scenarios(sc.M[None, :], sc.H[None, :], sc.eps_me[None, :])
    lp = np.zeros((len(grid), sc.M.shape[0]))
    for arch, n_a in zip(field.archetypes, counts):
        if n_a:
            lp += n_a * _cond_logF_logQ(grid[:, None], arch.exposure, contest, sc2)[0]
    cdf = np.exp(lp).mean(axis=1)
    return {q: float(np.expm1(np.interp(q, cdf, grid))) for q in qs}


# ---------- prize objective --------------------------------------------------
# Payout by final rank (MXN). Rules: only the single largest prize is paid per
# person, so across several accounts you maximise E[max(prizes)], not the sum.
PAYOUT_UNIVERSITY = (50_000, 20_000, 10_000)
PAYOUT_GENERAL = (100_000, 50_000, 10_000)


def expected_prize(r: RankResult, payout=PAYOUT_UNIVERSITY) -> float:
    """E[prize] = sum_k payout_k * P(rank == k)."""
    cum = np.concatenate([[0.0], r.p_top[: len(payout)]])
    return float(np.dot(payout, np.diff(cum)))
