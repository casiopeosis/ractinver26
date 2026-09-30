"""
Calibrate the field's risk-taking to a real observed winning score.

Anchor: Reto Actinver 2025, ~38-40k participants, winner +70.60% in six weeks.
One data point, so we fit ONE knob: a multiplier `lam` on every archetype's
idiosyncratic vol (how wild the field is), keeping the archetype mix fixed.
We match the *median* of the simulated winner's return to the observed winner.

Caveat worth remembering: last year's winner is a single draw from the winner's
distribution, and last year's market (IPC path) was one particular draw of M.
Treat the result as "right order of magnitude", not truth.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .engine import Scenarios, field_max_quantiles
from .model import Contest, Field


def scale_field(field: Field, lam: float) -> Field:
    return replace(field, archetypes=tuple(
        replace(a, exposure=replace(a.exposure, idio_vol=a.exposure.idio_vol * lam))
        for a in field.archetypes))


def calibrate_lambda(field: Field, contest: Contest, sc: Scenarios,
                     target_winner: float = 0.706, n_ref: int = 40_000,
                     lo: float = 0.05, hi: float = 2.0, iters: int = 18) -> float:
    ref = field.with_n(n_ref)
    grid = np.log1p(np.linspace(-0.3, 6.0, 316))
    f = lambda lam: field_max_quantiles(scale_field(ref, lam), contest, sc,
                                        qs=(0.5,), grid=grid)[0.5] - target_winner
    for _ in range(iters):          # median winner is increasing in lam -> bisection
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)
