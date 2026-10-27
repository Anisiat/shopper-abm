import numpy as np

from abm.agents import sample_agents
from abm.config import TRAITS, SimConfig


def test_traits_are_bounded_and_shaped() -> None:
    cfg = SimConfig()
    agents = sample_agents(cfg, np.random.default_rng(cfg.seed))

    assert agents.traits.shape == (cfg.n_agents, len(TRAITS))
    assert np.all((agents.traits >= 0) & (agents.traits <= 1))


def test_same_seed_gives_same_population() -> None:
    cfg = SimConfig()
    a = sample_agents(cfg, np.random.default_rng(cfg.seed))
    b = sample_agents(cfg, np.random.default_rng(cfg.seed))
    c = sample_agents(cfg, np.random.default_rng(cfg.seed + 1))

    assert np.array_equal(a.traits, b.traits)
    assert np.array_equal(a.archetype, b.archetype)
    assert not np.array_equal(a.traits, c.traits)
