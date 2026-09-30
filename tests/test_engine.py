import numpy as np
import pytest

from tournament import Archetype, Contest, Exposure, Field, default_field, draw_scenarios, rank_probabilities
from tournament.bruteforce import brute_force_rank


def test_symmetry_identical_players():
    """If you're statistically identical to all N rivals, P(win) = 1/(N+1)."""
    e = Exposure(b_M=1.0, b_H=0.5, idio_vol=0.3)
    n = 99
    field = Field(n, (Archetype("clone", 1.0, e),))
    c = Contest()
    sc = draw_scenarios(c, 40_000, seed=3)
    r = rank_probabilities(e, field, c, sc)
    assert abs(r.p_win - 1 / (n + 1)) < 4 * r.p_win_se
    assert r.p_top[4] == pytest.approx(5 / (n + 1), rel=0.05)


@pytest.mark.parametrize("me", [
    Exposure(b_M=1.0, idio_vol=0.05),
    Exposure(b_M=1.0, b_H=1.0, idio_vol=0.6),
    Exposure(b_M=1.5, idio_vol=1.0),
])
def test_matches_brute_force(me):
    field, c = default_field(200), Contest()
    sc = draw_scenarios(c, 20_000, seed=7)
    fast = rank_probabilities(me, field, c, sc)
    slow_win, slow_top = brute_force_rank(me, field, c, sc, seed=11)
    # brute force is noisy: allow ~4 binomial standard errors
    for k in (1, 5, 10):
        p_fast, p_slow = fast.p_top[k - 1], slow_top[k - 1]
        se = np.sqrt(max(p_slow * (1 - p_slow), 1e-6) / 20_000)
        assert abs(p_fast - p_slow) < 4 * se + 1e-3, (k, p_fast, p_slow)


def test_counts_sum_to_n():
    for n in (1, 7, 100, 503):
        assert default_field(n).counts().sum() == n
