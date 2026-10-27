"""All model settings in one place, so no number is hidden inside the code."""

from __future__ import annotations

from dataclasses import dataclass, field

from abm.products import ATTRIBUTES

TRAITS: tuple[str, ...] = (
    "price_sensitivity",
    "health_orientation",
    "novelty_seeking",
    "habit_strength",
)

MISSIONS: tuple[str, ...] = ("weekly_shop", "top_up", "treat")

# Utility terms a mission can scale: one per product attribute, plus habit.
WEIGHT_TERMS: tuple[str, ...] = ATTRIBUTES + ("habit",)


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
    mission_probs : tuple of float
        Weekly probability of each mission, ordered as ``MISSIONS``.
    """

    name: str
    share: float
    traits: dict[str, tuple[float, float]]
    mission_probs: tuple[float, ...] = (0.6, 0.3, 0.1)


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
            mission_probs=(0.4, 0.3, 0.3),     # treat occasions 3x more often
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


def default_mission_weights() -> dict[str, dict[str, float]]:
    """Return how each mission scales each utility term (assumed, not calibrated).

    A weight of 1.0 means "as on a normal weekly shop"; 0.3 means the term
    counts at 30% of its usual strength.

    Returns
    -------
    dict
        ``{mission: {term: weight}}`` for every mission in ``MISSIONS`` and
        every term in ``WEIGHT_TERMS``.
    """
    return {
        # Baseline: every term at full strength.
        "weekly_shop": {t: 1.0 for t in WEIGHT_TERMS},
        # Quick, low-effort trip: less browsing, more autopilot.
        "top_up": {
            "price": 1.0, "healthiness": 1.0, "novelty": 0.5,
            "brand_strength": 1.2, "treat_score": 1.0, "habit": 1.5,
        },
        # Treat occasion: price matters much less (the JD's example).
        "treat": {
            "price": 0.3, "healthiness": 0.5, "novelty": 1.2,
            "brand_strength": 1.0, "treat_score": 1.5, "habit": 0.8,
        },
    }


def default_beta() -> dict[str, float]:
    """Return the taste coefficients, one per utility term (assumed, not calibrated).

    Scaled so that within one shopper, products differ by roughly 1-3 utility
    units (1 unit is about 2.7x the odds). Price is per GBP and negative.

    Returns
    -------
    dict[str, float]
        ``{term: beta}`` for every term in ``WEIGHT_TERMS``.
    """
    return {
        "price": -3.0,
        "healthiness": 4.0,
        "novelty": 2.0,
        "brand_strength": 2.0,
        "treat_score": 3.0,
        "habit": 2.5,
    }


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
    mission_weights : dict
        ``{mission: {term: weight}}``, see ``default_mission_weights``.
    beta : dict
        Taste coefficients ``{term: beta}``, see ``default_beta``.
    outside_utility : float
        Utility of buying nothing. Fixed at 0 by convention; product
        utilities are measured relative to it.
    temperature : float
        Softmax temperature. 1.0 is the model; other values are only for
        sensitivity analysis (scale is not separately identified from beta).
    """

    n_agents: int = 500
    n_weeks: int = 26
    seed: int = 42
    launch_week: int = 4
    archetypes: tuple[ArchetypeSpec, ...] = field(default_factory=default_archetypes)
    mission_weights: dict[str, dict[str, float]] = field(
        default_factory=default_mission_weights
    )
    beta: dict[str, float] = field(default_factory=default_beta)
    outside_utility: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if set(self.beta) != set(WEIGHT_TERMS):
            raise ValueError(f"beta must define exactly {WEIGHT_TERMS}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        total = sum(a.share for a in self.archetypes)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Archetype shares must sum to 1, got {total}")
        for a in self.archetypes:
            if set(a.traits) != set(TRAITS):
                raise ValueError(f"Archetype {a.name!r} must define exactly {TRAITS}")
            if len(a.mission_probs) != len(MISSIONS) or abs(sum(a.mission_probs) - 1) > 1e-9:
                raise ValueError(f"Archetype {a.name!r} mission_probs must be "
                                 f"{len(MISSIONS)} values summing to 1")
        if set(self.mission_weights) != set(MISSIONS):
            raise ValueError(f"mission_weights must define exactly {MISSIONS}")
        for m, w in self.mission_weights.items():
            if set(w) != set(WEIGHT_TERMS):
                raise ValueError(f"Mission {m!r} must define exactly {WEIGHT_TERMS}")
