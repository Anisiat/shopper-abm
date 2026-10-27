"""Choice model: multinomial logit (utility + softmax + sampling).

Pipeline for one week::

    taste, habit_w = taste_coefficients(agents, W, cfg)    # who + why
    V = utility(taste, X, H, habit_w)                      # (N, J)
    P = choice_probabilities(V, available, cfg)            # (N, J + 1), last col = buy nothing
    choice = sample_choice(P, rng)                         # (N,), value J = buy nothing
"""

from __future__ import annotations

import numpy as np

from abm.agents import Agents
from abm.config import WEIGHT_TERMS, SimConfig

# Which trait scales which utility term. None = same for everyone.
TERM_TRAIT: dict[str, str | None] = {
    "price": "price_sensitivity",
    "healthiness": "health_orientation",
    "novelty": "novelty_seeking",
    "brand_strength": None,
    "treat_score": None,
    "habit": "habit_strength",
}


def taste_coefficients(
    agents: Agents, W: np.ndarray, cfg: SimConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Combine beta, traits and mission weights into per-agent coefficients.

    For agent ``i`` and term ``k``: ``coef_ik = beta_k * trait_ik * w_ik``,
    where ``trait_ik = 1`` for terms with no matching trait.

    Parameters
    ----------
    agents : Agents
        The population.
    W : np.ndarray, shape (n_agents, n_terms)
        Mission weights from ``mission_weight_matrix``.
    cfg : SimConfig
        Provides ``beta``.

    Returns
    -------
    taste : np.ndarray, shape (n_agents, n_attributes)
        Per-agent coefficient on each product attribute.
    habit_weight : np.ndarray, shape (n_agents,)
        Per-agent coefficient on the habit stock ``H``.
    """
    n = len(agents)
    coef = np.empty((n, len(WEIGHT_TERMS)))
    for k, term in enumerate(WEIGHT_TERMS):
        trait = TERM_TRAIT[term]
        scale = np.ones(n) if trait is None else agents.trait(trait)
        coef[:, k] = cfg.beta[term] * scale
    coef *= W
    h = WEIGHT_TERMS.index("habit")
    return np.delete(coef, h, axis=1), coef[:, h]


def utility(
    taste: np.ndarray, X: np.ndarray, H: np.ndarray, habit_weight: np.ndarray
) -> np.ndarray:
    """Systematic utility of every product for every agent.

    ``V_ij = sum_k taste_ik * X_jk + habit_weight_i * H_ij``

    Parameters
    ----------
    taste : np.ndarray, shape (n_agents, n_attributes)
    X : np.ndarray, shape (n_products, n_attributes)
    H : np.ndarray, shape (n_agents, n_products)
        Habit stock in [0, 1].
    habit_weight : np.ndarray, shape (n_agents,)

    Returns
    -------
    np.ndarray, shape (n_agents, n_products)
    """
    return taste @ X.T + habit_weight[:, None] * H


def softmax(V: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax, numerically stable.

    ``P_ij = exp(V_ij / T) / sum_l exp(V_il / T)``

    Must not overflow for large utilities (e.g. 1000) and must give
    probability 0 to entries equal to ``-np.inf``.

    Parameters
    ----------
    V : np.ndarray, shape (n_agents, n_options)
    temperature : float
        Higher = more random choices, lower = closer to always picking the max.

    Returns
    -------
    np.ndarray, shape (n_agents, n_options)
        Each row sums to 1.
    """
    z = V / temperature
    z = z - z.max(axis=1, keepdims=True)   # shift so the largest entry is 0: no overflow
    e = np.exp(z)                          # exp(-inf) = 0, so excluded options stay at 0
    return e / e.sum(axis=1, keepdims=True)


def choice_probabilities(
    V: np.ndarray, available: np.ndarray, cfg: SimConfig
) -> np.ndarray:
    """Choice probabilities over products plus a "buy nothing" option.

    Unavailable products must get probability 0. The outside option
    (utility ``cfg.outside_utility``) is appended as the **last** column.

    Parameters
    ----------
    V : np.ndarray, shape (n_agents, n_products)
    available : np.ndarray of bool, shape (n_products,)
    cfg : SimConfig
        Provides ``outside_utility`` and ``temperature``.

    Returns
    -------
    np.ndarray, shape (n_agents, n_products + 1)
    """
    V = np.where(available[None, :], V, -np.inf)
    outside = np.full((V.shape[0], 1), cfg.outside_utility)
    return softmax(np.hstack([V, outside]), cfg.temperature)


def sample_choice(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one option per agent from its row of probabilities.

    Parameters
    ----------
    P : np.ndarray, shape (n_agents, n_options)
        Each row sums to 1.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of int, shape (n_agents,)
        Chosen column index per agent.
    """
    u = rng.random((P.shape[0], 1))
    # >= (not >) so a draw of exactly 0 cannot select a leading zero-probability option.
    idx = (u >= P.cumsum(axis=1)).sum(axis=1)
    # Guard against floating-point rounding where a row's cumsum ends at 0.9999999.
    return np.minimum(idx, P.shape[1] - 1)
