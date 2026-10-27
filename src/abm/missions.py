"""Intentions: each week every shopper draws a mission that reweights their utility."""

from __future__ import annotations

import numpy as np

from abm.agents import Agents
from abm.config import MISSIONS, WEIGHT_TERMS, SimConfig


def mission_probabilities(agents: Agents, cfg: SimConfig) -> np.ndarray:
    """Look up each agent's mission probabilities from its archetype.

    Parameters
    ----------
    agents : Agents
        The population.
    cfg : SimConfig
        Provides ``mission_probs`` per archetype.

    Returns
    -------
    np.ndarray, shape (n_agents, n_missions)
        Row ``i`` holds agent ``i``'s probabilities, ordered as ``MISSIONS``.
    """
    table = {a.name: a.mission_probs for a in cfg.archetypes}
    return np.array([table[name] for name in agents.archetype])


def draw_missions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one mission per agent using the inverse-CDF trick.

    For each agent, draw ``u ~ Uniform(0, 1)`` and pick the first mission
    whose cumulative probability exceeds ``u``. Doing this for all rows at
    once avoids a Python loop over agents.

    Parameters
    ----------
    probs : np.ndarray, shape (n_agents, n_missions)
        Each row sums to 1.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of int, shape (n_agents,)
        Mission index per agent (index into ``MISSIONS``).
    """
    u = rng.random((probs.shape[0], 1))
    idx = (u > probs.cumsum(axis=1)).sum(axis=1)
    # Guard against floating-point rounding where a row's cumsum ends at 0.9999999.
    return np.minimum(idx, probs.shape[1] - 1)


def mission_weight_matrix(missions: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Turn mission indices into per-agent utility weights.

    Parameters
    ----------
    missions : np.ndarray of int, shape (n_agents,)
        Output of ``draw_missions``.
    cfg : SimConfig
        Provides ``mission_weights``.

    Returns
    -------
    np.ndarray, shape (n_agents, n_terms)
        Weight for each utility term, columns ordered as ``WEIGHT_TERMS``.
    """
    table = np.array(
        [[cfg.mission_weights[m][t] for t in WEIGHT_TERMS] for m in MISSIONS]
    )
    return table[missions]  # fancy indexing: one table row per agent
