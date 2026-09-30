"""
Brute-force Monte Carlo: simulate every rival explicitly.

Slow and noisy, which is exactly why it exists: it makes no clever
assumptions, so it's the referee that checks engine.py (see tests/).
"""
from __future__ import annotations

import numpy as np

from .engine import Scenarios, log_return, unit_t_sample
from .model import Contest, Exposure, Field


def brute_force_rank(me: Exposure, field: Field, contest: Contest, sc: Scenarios,
                     seed: int = 1, k_max: int = 10, chunk: int = 2000) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = log_return(me, contest, sc, sc.eps_me)
    S = x.shape[0]
    ahead = np.zeros(S, dtype=int)
    for arch, n_a in zip(field.archetypes, field.counts()):
        for lo in range(0, S, chunk):
            sl = slice(lo, min(lo + chunk, S))
            sub = Scenarios(sc.M[sl], sc.H[sl], sc.eps_me[sl])
            eps = unit_t_sample(rng, arch.exposure.df, (n_a, sub.M.shape[0]))
            r = log_return(arch.exposure, contest, sub, eps)          # (n_a, chunk)
            ahead[sl] += (r > x[sl]).sum(axis=0)
    p_top = np.array([(ahead < k).mean() for k in range(1, k_max + 1)])
    return float(p_top[0]), p_top
