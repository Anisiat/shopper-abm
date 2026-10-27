import numpy as np

from abm.config import MISSIONS, WEIGHT_TERMS, SimConfig
from abm.missions import draw_missions, mission_weight_matrix


def test_draw_matches_probabilities_on_average() -> None:
    # Law of large numbers: with many draws, frequencies approach probabilities.
    rng = np.random.default_rng(0)
    p = np.array([0.6, 0.3, 0.1])
    probs = np.tile(p, (100_000, 1))
    freq = np.bincount(draw_missions(probs, rng), minlength=3) / len(probs)
    assert np.allclose(freq, p, atol=0.01)


def test_certain_mission_is_always_drawn() -> None:
    rng = np.random.default_rng(0)
    probs = np.tile([0.0, 0.0, 1.0], (50, 1))
    assert np.all(draw_missions(probs, rng) == 2)


def test_treat_mission_downweights_price() -> None:
    cfg = SimConfig()
    missions = np.array([MISSIONS.index("weekly_shop"), MISSIONS.index("treat")])
    W = mission_weight_matrix(missions, cfg)
    price = WEIGHT_TERMS.index("price")

    assert W.shape == (2, len(WEIGHT_TERMS))
    assert W[0, price] == 1.0
    assert W[1, price] < W[0, price]
