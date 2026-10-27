import numpy as np
import pytest

from abm.agents import sample_agents
from abm.config import MISSIONS, SimConfig
from abm.choice import (
    choice_probabilities,
    sample_choice,
    softmax,
    taste_coefficients,
    utility,
)
from abm.missions import mission_weight_matrix
from abm.products import ATTRIBUTES


# ---------- taste_coefficients (already implemented) ----------

def test_treat_mission_softens_price_coefficient() -> None:
    cfg = SimConfig()
    agents = sample_agents(cfg, np.random.default_rng(0))
    n = len(agents)
    price = ATTRIBUTES.index("price")

    weekly = mission_weight_matrix(np.full(n, MISSIONS.index("weekly_shop")), cfg)
    treat = mission_weight_matrix(np.full(n, MISSIONS.index("treat")), cfg)
    taste_w, _ = taste_coefficients(agents, weekly, cfg)
    taste_t, _ = taste_coefficients(agents, treat, cfg)

    assert np.all(taste_w[:, price] < 0)                       # price is a cost
    assert np.allclose(taste_t[:, price], 0.3 * taste_w[:, price])


# ---------- utility ----------

def test_utility_hand_computed() -> None:
    taste = np.array([[1.0, 0.0],
                      [0.0, 2.0]])          # 2 agents, 2 attributes
    X = np.array([[1.0, 1.0],
                  [2.0, 0.0],
                  [0.0, 3.0]])              # 3 products
    H = np.array([[0.0, 1.0, 0.0],
                  [0.5, 0.0, 0.0]])
    habit_weight = np.array([10.0, 2.0])

    expected = np.array([[1.0, 2.0 + 10.0, 0.0],
                         [2.0 + 1.0, 0.0, 6.0]])
    assert np.allclose(utility(taste, X, H, habit_weight), expected)


# ---------- softmax ----------

def test_softmax_rows_sum_to_one() -> None:
    V = np.random.default_rng(0).normal(size=(20, 5))
    P = softmax(V)
    assert P.shape == V.shape
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)


def test_softmax_only_differences_matter() -> None:
    V = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(V), softmax(V + 100.0))


def test_softmax_is_numerically_stable() -> None:
    P = softmax(np.array([[1000.0, 1001.0]]))
    assert np.all(np.isfinite(P))
    assert np.allclose(P, [[1 / (1 + np.e), np.e / (1 + np.e)]])


def test_softmax_temperature() -> None:
    V = np.array([[0.0, 1.0]])
    assert softmax(V, temperature=0.01)[0, 1] > 0.99      # nearly deterministic
    assert np.allclose(softmax(V, temperature=1000.0), 0.5, atol=1e-3)  # nearly uniform


def test_softmax_minus_inf_gets_zero() -> None:
    P = softmax(np.array([[0.0, -np.inf, 0.0]]))
    assert np.allclose(P, [[0.5, 0.0, 0.5]])


# ---------- choice_probabilities ----------

def test_choice_probabilities_outside_option_and_availability() -> None:
    cfg = SimConfig()                                  # outside_utility = 0
    V = np.zeros((4, 3))
    available = np.array([True, False, True])
    P = choice_probabilities(V, available, cfg)

    assert P.shape == (4, 4)                           # 3 products + buy nothing
    assert np.allclose(P[:, 1], 0.0)                   # unavailable
    assert np.allclose(P[:, [0, 2, 3]], 1 / 3)         # equal utilities
    assert np.allclose(P.sum(axis=1), 1.0)


# ---------- sample_choice ----------

def test_sample_choice_frequencies() -> None:
    rng = np.random.default_rng(0)
    p = np.array([0.2, 0.5, 0.0, 0.3])
    P = np.tile(p, (100_000, 1))
    freq = np.bincount(sample_choice(P, rng), minlength=4) / len(P)
    assert np.allclose(freq, p, atol=0.01)


def test_sample_choice_never_picks_zero_probability() -> None:
    rng = np.random.default_rng(0)
    P = np.tile([0.0, 1.0, 0.0], (1000, 1))
    assert np.all(sample_choice(P, rng) == 1)
