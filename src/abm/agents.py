"""The synthetic population: shoppers sampled from archetypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from abm.config import TRAITS, SimConfig


@dataclass(frozen=True)
class Agents:
    """A population of shoppers stored as arrays (one row per agent).

    Parameters
    ----------
    archetype : np.ndarray of str, shape (n_agents,)
        Archetype label of each agent.
    traits : np.ndarray, shape (n_agents, n_traits)
        Trait values in [0, 1], columns ordered as ``TRAITS``.
    """

    archetype: np.ndarray
    traits: np.ndarray

    def __len__(self) -> int:
        return len(self.archetype)

    def trait(self, name: str) -> np.ndarray:
        """Return one trait column, e.g. ``agents.trait("price_sensitivity")``."""
        return self.traits[:, TRAITS.index(name)]

    def to_frame(self) -> pd.DataFrame:
        """Return the population as a DataFrame (one row per agent)."""
        df = pd.DataFrame(self.traits, columns=list(TRAITS))
        df.insert(0, "archetype", self.archetype)
        return df


def sample_agents(cfg: SimConfig, rng: np.random.Generator) -> Agents:
    """Sample a synthetic population from the archetype specs in ``cfg``.

    Each agent first draws an archetype with probability equal to its share,
    then each trait is drawn from that archetype's Beta(a, b) distribution.

    Parameters
    ----------
    cfg : SimConfig
        Provides ``n_agents`` and ``archetypes``.
    rng : np.random.Generator
        Random number generator (create with ``np.random.default_rng(seed)``).

    Returns
    -------
    Agents
        The sampled population.
    """
    names = np.array([a.name for a in cfg.archetypes])
    shares = np.array([a.share for a in cfg.archetypes])
    idx = rng.choice(len(names), size=cfg.n_agents, p=shares)

    traits = np.empty((cfg.n_agents, len(TRAITS)))
    for k, spec in enumerate(cfg.archetypes):
        rows = idx == k
        for t, trait in enumerate(TRAITS):
            a, b = spec.traits[trait]
            traits[rows, t] = rng.beta(a, b, size=rows.sum())

    return Agents(archetype=names[idx], traits=traits)
