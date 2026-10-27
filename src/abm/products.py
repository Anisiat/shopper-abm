"""The shopping environment: products on the shelf and their attributes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ATTRIBUTES: tuple[str, ...] = (
    "price",
    "healthiness",
    "novelty",
    "brand_strength",
    "treat_score",
)


@dataclass(frozen=True)
class Product:
    """One product in the category (chilled yoghurts).

    Parameters
    ----------
    name : str
        Display name (fictional, no real brands).
    price : float
        Shelf price in GBP.
    healthiness, novelty, brand_strength, treat_score : float
        Attribute scores on a 0-1 scale.
    launch_week : int
        First week the product is on sale (0 = always available).
    """

    name: str
    price: float
    healthiness: float
    novelty: float
    brand_strength: float
    treat_score: float
    launch_week: int = 0


def default_catalogue(launch_week: int = 4) -> list[Product]:
    """Return the six default products (attribute values are assumptions).

    Parameters
    ----------
    launch_week : int
        Week in which the new kefir product launches.

    Returns
    -------
    list of Product
        Five established products plus one new launch.
    """
    return [
        Product("Value Fruit Pot",      0.70, 0.30, 0.10, 0.10, 0.40),
        Product("Own-label Natural",    0.90, 0.60, 0.10, 0.20, 0.20),
        Product("Classic Brand Fruit",  1.60, 0.40, 0.20, 0.80, 0.50),
        Product("Greek Protein Pot",    2.20, 0.90, 0.40, 0.60, 0.20),
        Product("Indulgent Dessert",    2.50, 0.10, 0.30, 0.70, 0.95),
        Product("Live Kefir (NEW)",     1.90, 0.85, 0.95, 0.30, 0.30, launch_week),
    ]


def attribute_matrix(
    products: list[Product], week: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build the attribute matrix and availability mask for a given week.

    Parameters
    ----------
    products : list of Product
        The catalogue.
    week : int
        Current simulation week.

    Returns
    -------
    X : np.ndarray, shape (n_products, n_attributes)
        Attribute values, columns ordered as ``ATTRIBUTES``.
    available : np.ndarray of bool, shape (n_products,)
        True where the product is on the shelf this week.
    """
    X = np.array([[getattr(p, a) for a in ATTRIBUTES] for p in products], dtype=float)
    available = np.array([week >= p.launch_week for p in products])
    return X, available


def products_frame(products: list[Product]) -> pd.DataFrame:
    """Return the catalogue as a DataFrame, for inspection and plotting.

    Parameters
    ----------
    products : list of Product
        The catalogue.

    Returns
    -------
    pd.DataFrame
        One row per product, indexed by name.
    """
    return pd.DataFrame([asdict(p) for p in products]).set_index("name")
