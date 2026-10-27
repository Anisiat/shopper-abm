"""Extraneous factors: outside events that change the shelf or what people value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abm.products import ATTRIBUTES, Product


@dataclass(frozen=True)
class PriceChange:
    """A multiplicative price change on one product.

    Parameters
    ----------
    product : str
        Product name, exactly as in the catalogue.
    start : int
        First week the change applies.
    end : int or None
        Last week it applies (inclusive). ``None`` means it never ends.
    factor : float
        Price multiplier, e.g. 0.75 for 25% off or 1.2 for a 20% rise.
    """

    product: str
    start: int
    end: int | None
    factor: float


@dataclass(frozen=True)
class HealthTrend:
    """A shift in how much everyone cares about healthiness.

    Parameters
    ----------
    start : int
        First week of the trend.
    multiplier : float
        Factor applied to every shopper's healthiness coefficient from
        ``start`` onwards (1.5 = health matters 50% more).
    """

    start: int
    multiplier: float


@dataclass(frozen=True)
class Scenario:
    """A named set of outside events.

    Parameters
    ----------
    name : str
        Label used in tables and plots.
    price_changes : tuple of PriceChange
        Promotions and price rises.
    health_trend : HealthTrend or None
        Optional cultural shift towards healthier products.
    """

    name: str
    price_changes: tuple[PriceChange, ...] = ()
    health_trend: HealthTrend | None = None


BASELINE = Scenario("baseline")


def apply_price_changes(
    scenario: Scenario, products: list[Product], week: int, X: np.ndarray
) -> np.ndarray:
    """Return the attribute matrix with this week's price changes applied.

    Parameters
    ----------
    scenario : Scenario
        The events to apply.
    products : list of Product
        The catalogue, used to find each product's row.
    week : int
        Current simulation week.
    X : np.ndarray, shape (n_products, n_attributes)
        Attribute matrix from ``attribute_matrix``. Not modified.

    Returns
    -------
    np.ndarray, shape (n_products, n_attributes)
        A copy of ``X`` with the ``price`` column adjusted.
    """
    names = [p.name for p in products]
    price = ATTRIBUTES.index("price")
    X = X.copy()
    for change in scenario.price_changes:
        if change.product not in names:
            raise ValueError(f"Unknown product {change.product!r}")
        active = week >= change.start and (change.end is None or week <= change.end)
        if active:
            X[names.index(change.product), price] *= change.factor
    return X


def health_multiplier(scenario: Scenario, week: int) -> float:
    """Return the factor on the healthiness coefficient for this week.

    Parameters
    ----------
    scenario : Scenario
        The events to apply.
    week : int
        Current simulation week.

    Returns
    -------
    float
        ``1.0`` when no trend is active.
    """
    trend = scenario.health_trend
    if trend is not None and week >= trend.start:
        return trend.multiplier
    return 1.0


def default_scenarios() -> dict[str, Scenario]:
    """Return the preset scenarios used in the demo (all assumptions).

    Returns
    -------
    dict[str, Scenario]
        Baseline plus a promotion on the new kefir, a price rise on the
        leading brand, and a health trend.
    """
    return {
        "baseline": BASELINE,
        "kefir_promo": Scenario(
            "kefir_promo", price_changes=(PriceChange("Live Kefir (NEW)", 8, 11, 0.75),)
        ),
        "brand_price_rise": Scenario(
            "brand_price_rise",
            price_changes=(PriceChange("Classic Brand Fruit", 12, None, 1.25),),
        ),
        "health_trend": Scenario("health_trend", health_trend=HealthTrend(10, 1.5)),
    }
