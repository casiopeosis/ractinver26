"""
Return model for the tournament.

Every portfolio (yours or a rival's) is described by a linear factor model over
the contest horizon T (in years):

    log R = m + b_M * M + b_H * H + s * eps

    M    market factor   (the IPC / broad BMV), common to everyone
    H    "hot basket"    the popular names the crowd piles into, independent of M
    eps  idiosyncratic   unit-variance Student-t, independent across people

Conditional on (M, H), rivals are independent draws -> that is what lets
engine.py compute rank probabilities exactly instead of simulating every rival.

Inputs are annualised (vol, drift) because that's how everyone thinks about
them; they are scaled to the horizon here. The log-mean includes the Ito
correction  -0.5 * total_var * T, so cranking volatility has a real cost in
median outcome (volatility drag). Rankings use R, which is monotone in log R.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Factors:
    mu_M: float = 0.08      # annual expected arithmetic return of the market
    vol_M: float = 0.18     # annual vol of the market (IPC ~ 15-20%)
    vol_H: float = 0.35     # annual vol of the hot basket, net of market
    df: float = 4.0         # tail heaviness of factor shocks (inf -> Gaussian)


@dataclass(frozen=True)
class Exposure:
    """A portfolio's loadings. Shared by rival archetypes and your strategy."""
    b_M: float = 1.0        # market beta (>1 only via leveraged ETFs / margin)
    b_H: float = 0.0        # loading on the crowd's hot basket
    idio_vol: float = 0.0   # annual idiosyncratic vol (concentration / trading noise)
    alpha: float = 0.0      # annual arithmetic excess return (skill minus costs)
    df: float = 4.0         # tail heaviness of the idiosyncratic shock

    def total_var_annual(self, f: Factors) -> float:
        return (self.b_M * f.vol_M) ** 2 + (self.b_H * f.vol_H) ** 2 + self.idio_vol ** 2

    def log_mean(self, f: Factors, T: float) -> float:
        mu = self.b_M * f.mu_M + self.alpha
        return (mu - 0.5 * self.total_var_annual(f)) * T


@dataclass(frozen=True)
class Archetype:
    name: str
    share: float            # fraction of the field
    exposure: Exposure


@dataclass(frozen=True)
class Field:
    n: int                              # number of rivals (excluding you)
    archetypes: tuple[Archetype, ...]

    def counts(self) -> np.ndarray:
        """Integer head-count per archetype, summing exactly to n."""
        shares = np.array([a.share for a in self.archetypes], dtype=float)
        shares = shares / shares.sum()
        raw = shares * self.n
        c = np.floor(raw).astype(int)
        # hand leftover seats to the largest remainders
        for i in np.argsort(raw - c)[::-1][: self.n - c.sum()]:
            c[i] += 1
        return c

    def with_n(self, n: int) -> "Field":
        return replace(self, n=n)


@dataclass(frozen=True)
class Contest:
    trading_days: int = 30      # Oct 5 - Nov 13, 2026
    factors: Factors = field(default_factory=Factors)
    commission: float = 0.00116 # per side: 0.10% + 16% IVA (confirmed on the platform)

    def cost_drag(self, round_trips: float) -> float:
        """Annualised alpha lost to commissions for a given number of full
        portfolio round trips over the contest. Subtract from Exposure.alpha."""
        return 2 * self.commission * round_trips / self.T

    @property
    def T(self) -> float:
        return self.trading_days / TRADING_DAYS_PER_YEAR


# --------------------------------------------------------------------------
# Default field: a guess at a university cohort. EVERY number here is a prior,
# meant to be overwritten once you see the practice-week leaderboard.
# --------------------------------------------------------------------------
def default_field(n: int = 500) -> Field:
    return Field(
        n=n,
        archetypes=(
            # signed up, bought a few blue chips (or stayed in cash), forgot
            Archetype("dormant", 0.35, Exposure(b_M=0.5, idio_vol=0.08)),
            # diversified ETF / index holder
            Archetype("indexer", 0.20, Exposure(b_M=1.0, idio_vol=0.04)),
            # 5-10 popular names everyone talks about -> correlated via H
            Archetype("hot_chaser", 0.25, Exposure(b_M=1.0, b_H=1.0, idio_vol=0.25)),
            # trades a lot; noise + costs ~ negative alpha
            Archetype("active", 0.15, Exposure(b_M=0.9, b_H=0.3, idio_vol=0.50, alpha=-0.10)),
            # swings for the fences: concentrated, leveraged if allowed
            Archetype("gambler", 0.05, Exposure(b_M=1.5, b_H=0.5, idio_vol=0.90, alpha=-0.20)),
        ),
    )
