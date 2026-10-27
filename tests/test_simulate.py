import numpy as np

from abm.config import SimConfig
from abm.simulate import run_simulation, update_habit


# ---------- update_habit (yours to implement) ----------

def test_bought_product_moves_toward_one() -> None:
    H = np.array([[0.0, 0.5]])
    new = update_habit(H, np.array([0]), alpha=0.3)
    assert np.allclose(new[0, 0], 0.3)          # 0 + 0.3 * (1 - 0)


def test_unbought_products_decay_toward_zero() -> None:
    H = np.array([[0.0, 0.5]])
    new = update_habit(H, np.array([0]), alpha=0.3)
    assert np.allclose(new[0, 1], 0.35)         # 0.5 + 0.3 * (0 - 0.5)


def test_buying_nothing_decays_every_habit() -> None:
    H = np.array([[0.8, 0.4]])
    new = update_habit(H, np.array([2]), alpha=0.5)   # index 2 = "nothing" (2 products)
    assert np.allclose(new, [[0.4, 0.2]])


def test_each_agent_updated_independently() -> None:
    H = np.zeros((2, 2))
    new = update_habit(H, np.array([0, 1]), alpha=1.0)
    assert np.allclose(new, np.eye(2))


def test_habit_stays_in_unit_interval_and_input_unchanged() -> None:
    rng = np.random.default_rng(0)
    H = rng.random((50, 6))
    H_before = H.copy()
    new = update_habit(H, rng.integers(0, 7, size=50), alpha=0.3)
    assert np.all((new >= 0) & (new <= 1))
    assert np.array_equal(H, H_before)


def test_five_purchases_in_a_row_build_strong_habit() -> None:
    H = np.zeros((1, 3))
    for _ in range(5):
        H = update_habit(H, np.array([1]), alpha=0.3)
    assert np.isclose(H[0, 1], 1 - 0.7**5)       # about 0.83


# ---------- run_simulation (needs update_habit) ----------

def test_output_shapes_and_valid_choices() -> None:
    cfg = SimConfig(n_agents=100, n_weeks=8)
    res = run_simulation(cfg)
    assert res.choices.shape == (8, 100)
    assert res.missions.shape == (8, 100)
    assert res.choices.min() >= 0 and res.choices.max() <= len(res.products)


def test_new_product_not_bought_before_launch() -> None:
    cfg = SimConfig(n_agents=300, n_weeks=10, launch_week=4)
    res = run_simulation(cfg)
    new = len(res.products) - 1                  # the kefir is the last product
    assert not np.any(res.choices[:4] == new)
    assert np.any(res.choices[4:] == new)


def test_same_seed_same_run() -> None:
    cfg = SimConfig(n_agents=100, n_weeks=6)
    a, b = run_simulation(cfg), run_simulation(cfg)
    assert np.array_equal(a.choices, b.choices)
