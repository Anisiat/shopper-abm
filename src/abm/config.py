"""All model settings in one place, so no number is hidden inside the code."""

from __future__ import annotations

from dataclasses import dataclass, field

TRAITS: tuple[str, ...] = (
    "price_sensitivity",
    "health_orientation",
    "novelty_seeking",
    "habit_strength",
)


@dataclass(frozen=True)
class ArchetypeSpec:
    """One shopper archetype: its population share and trait distributions.

    Parameters
    ----------
    name : str
        Archetype label, e.g. ``"budget"``.
    share : float
        Fraction of the population drawn from this archetype.
    traits : dict[str, tuple[float, float]]
        For each trait in ``TRAITS``, the ``(a, b)`` parameters of a Beta
        distribution. The mean is ``a / (a + b)``; larger ``a + b`` means
        less spread between people of the same archetype.
    """

    name: str
    share: float
    traits: dict[str, tuple[float, float]]


def default_archetypes() -> tuple[ArchetypeSpec, ...]:
    """Return the four default archetypes (assumed, not calibrated).

    Returns
    -------
    tuple of ArchetypeSpec
        Budget, health-focused, treat seeker and habitual shoppers.
    """
    return (
        ArchetypeSpec(
            "budget",
            0.30,
            {
                "price_sensitivity": (8, 2),   # mean 0.80
                "health_orientation": (2, 5),  # mean 0.29
                "novelty_seeking": (2, 6),     # mean 0.25
                "habit_strength": (4, 4),      # mean 0.50
            },
        ),
        ArchetypeSpec(
            "health_focused",
            0.25,
            {
                "price_sensitivity": (3, 5),   # mean 0.38
                "health_orientation": (8, 2),  # mean 0.80
                "novelty_seeking": (4, 4),     # mean 0.50
                "habit_strength": (4, 4),      # mean 0.50
            },
        ),
        ArchetypeSpec(
            "treat_seeker",
            0.20,
            {
                "price_sensitivity": (3, 5),   # mean 0.38
                "health_orientation": (2, 6),  # mean 0.25
                "novelty_seeking": (5, 3),     # mean 0.63
                "habit_strength": (3, 5),      # mean 0.38
            },
        ),
        ArchetypeSpec(
            "habitual",
            0.25,
            {
                "price_sensitivity": (4, 4),   # mean 0.50
                "health_orientation": (4, 4),  # mean 0.50
                "novelty_seeking": (2, 8),     # mean 0.20
                "habit_strength": (8, 2),      # mean 0.80
            },
        ),
    )


@dataclass(frozen=True)
class SimConfig:
    """Settings for one simulation run.

    Parameters
    ----------
    n_agents : int
        Number of synthetic shoppers.
    n_weeks : int
        Number of weekly time steps (weeks are numbered from 0).
    seed : int
        Seed for ``np.random.default_rng``; same seed gives the same run.
    launch_week : int
        Week in which the new product first appears on the shelf.
    archetypes : tuple of ArchetypeSpec
        Shopper types to sample from. Shares must sum to 1.
    """

    n_agents: int = 500
    n_weeks: int = 26
    seed: int = 42
    launch_week: int = 4
    archetypes: tuple[ArchetypeSpec, ...] = field(default_factory=default_archetypes)

    def __post_init__(self) -> None:
        total = sum(a.share for a in self.archetypes)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Archetype shares must sum to 1, got {total}")
        for a in self.archetypes:
            if set(a.traits) != set(TRAITS):
                raise ValueError(f"Archetype {a.name!r} must define exactly {TRAITS}")
