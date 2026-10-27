"""The time loop: weekly missions, choices and habit updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abm.agents import Agents, sample_agents
from abm.choice import (
    choice_probabilities,
    sample_choice,
    taste_coefficients,
    utility,
)
from abm.config import SimConfig
from abm.missions import draw_missions, mission_probabilities, mission_weight_matrix
from abm.products import ATTRIBUTES, Product, attribute_matrix, default_catalogue
from abm.scenarios import BASELINE, Scenario, apply_price_changes, health_multiplier


@dataclass(frozen=True)
class SimResult:
    """Everything recorded during one run.

    Parameters
    ----------
    choices : np.ndarray of int, shape (n_weeks, n_agents)
        Chosen product index per agent per week; ``n_products`` means
        "bought nothing".
    missions : np.ndarray of int, shape (n_weeks, n_agents)
        Mission index per agent per week (index into ``MISSIONS``).
    habit_final : np.ndarray, shape (n_agents, n_products)
        Habit stock at the end of the run.
    agents : Agents
        The simulated population.
    products : list of Product
        The catalogue.
    """

    choices: np.ndarray
    missions: np.ndarray
    habit_final: np.ndarray
    agents: Agents
    products: list[Product]


def update_habit(H: np.ndarray, choice: np.ndarray, alpha: float) -> np.ndarray:
    """Update every agent's habit stock after one week of shopping.

    Delta rule: ``H_ij <- H_ij + alpha * (bought_ij - H_ij)``, where
    ``bought_ij`` is 1 if agent ``i`` bought product ``j`` this week, else 0.
    An agent who bought nothing has ``bought_ij = 0`` for every product.

    Parameters
    ----------
    H : np.ndarray, shape (n_agents, n_products)
        Habit stock in [0, 1] before this week's purchase.
    choice : np.ndarray of int, shape (n_agents,)
        Chosen column per agent; the value ``n_products`` means "bought nothing".
    alpha : float
        Learning rate in (0, 1].

    Returns
    -------
    np.ndarray, shape (n_agents, n_products)
        New habit stock. The input array must not be modified.
    """
    n_agents, n_products = H.shape
    # One-hot with an extra column for "bought nothing", then drop that column
    # so a no-purchase week gives a row of all zeros.
    bought = np.zeros((n_agents, n_products + 1))
    bought[np.arange(n_agents), choice] = 1.0
    bought = bought[:, :n_products]
    return H + alpha * (bought - H)   # new array: H itself is not modified


def run_simulation(cfg: SimConfig, scenario: Scenario = BASELINE) -> SimResult:
    """Run one simulation of ``cfg.n_weeks`` weekly steps.

    Parameters
    ----------
    cfg : SimConfig
        All model settings, including the random seed.
    scenario : Scenario
        Outside events (price changes, health trend). Defaults to none.
        The same seed gives the same random draws in every scenario, so
        runs differ only because of the scenario (common random numbers).

    Returns
    -------
    SimResult
        The full purchase log and final state.
    """
    rng = np.random.default_rng(cfg.seed)
    agents = sample_agents(cfg, rng)
    products = default_catalogue(cfg.launch_week)
    n, j = len(agents), len(products)

    mission_probs = mission_probabilities(agents, cfg)
    H = np.zeros((n, j))
    choices = np.empty((cfg.n_weeks, n), dtype=int)
    missions = np.empty((cfg.n_weeks, n), dtype=int)

    for week in range(cfg.n_weeks):
        X, available = attribute_matrix(products, week)
        X = apply_price_changes(scenario, products, week, X)

        missions[week] = draw_missions(mission_probs, rng)
        W = mission_weight_matrix(missions[week], cfg)

        taste, habit_weight = taste_coefficients(agents, W, cfg)
        taste[:, ATTRIBUTES.index("healthiness")] *= health_multiplier(scenario, week)
        V = utility(taste, X, H, habit_weight)
        P = choice_probabilities(V, available, cfg)
        chosen = sample_choice(P, rng)
        shops = rng.random(n) < cfg.category_prob          # category gate
        choices[week] = np.where(shops, chosen, j)         # j = bought nothing

        H = update_habit(H, choices[week], cfg.habit_rate)

    return SimResult(choices, missions, H, agents, products)
