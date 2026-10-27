"""Sensitivity analysis: how much do outputs move when an assumption changes?"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from abm.config import SimConfig
from abm.metrics import market_share, repeat_metrics
from abm.scenarios import default_scenarios
from abm.simulate import run_simulation

KEFIR = "Live Kefir (NEW)"
BRAND = "Classic Brand Fruit"

Metric = Callable[[SimConfig], float]

_PLAIN_PARAMS = {"habit_rate", "temperature", "category_prob", "outside_utility"}


def with_param(cfg: SimConfig, name: str, value: float) -> SimConfig:
    """Return a copy of ``cfg`` with one parameter changed.

    Parameters
    ----------
    cfg : SimConfig
        The base configuration. Not modified.
    name : str
        Either ``"beta.<term>"`` (e.g. ``"beta.habit"``) or one of
        ``habit_rate``, ``temperature``, ``category_prob``, ``outside_utility``.
    value : float
        New value.

    Returns
    -------
    SimConfig
        The modified copy.
    """
    if name.startswith("beta."):
        term = name.removeprefix("beta.")
        if term not in cfg.beta:
            raise ValueError(f"Unknown beta term {term!r}")
        return replace(cfg, beta={**cfg.beta, term: value})
    if name in _PLAIN_PARAMS:
        return replace(cfg, **{name: value})
    raise ValueError(f"Unknown parameter {name!r}")


def kefir_trial_rate(cfg: SimConfig) -> float:
    """Share of shoppers who ever buy the new kefir (0-1)."""
    return repeat_metrics(run_simulation(cfg), KEFIR)["trial_rate"]


def kefir_repeat_rate(cfg: SimConfig) -> float:
    """Share of observable kefir triers who buy it again within 8 weeks (0-1)."""
    return repeat_metrics(run_simulation(cfg), KEFIR)["repeat_rate"]


def brand_price_rise_effect(cfg: SimConfig) -> float:
    """Change in Classic Brand's share of purchases after its price rise.

    Compares the ``brand_price_rise`` scenario with baseline (same seed) over
    the weeks the rise is active.

    Returns
    -------
    float
        Difference in percentage points (negative when the rise loses share).
    """
    scenario = default_scenarios()["brand_price_rise"]
    start = scenario.price_changes[0].start
    base = market_share(run_simulation(cfg))[BRAND]
    rise = market_share(run_simulation(cfg, scenario))[BRAND]
    return float((rise[start:].mean() - base[start:].mean()) * 100)


def run_over_seeds(cfg: SimConfig, metric: Metric, seeds: Iterable[int]) -> np.ndarray:
    """Evaluate ``metric`` once per random seed (Monte Carlo repetitions).

    Parameters
    ----------
    cfg : SimConfig
        Base configuration; only the seed changes.
    metric : callable
        Function from a config to one number.
    seeds : iterable of int
        Seeds to use.

    Returns
    -------
    np.ndarray
        One value per seed.
    """
    return np.array([metric(replace(cfg, seed=s)) for s in seeds], dtype=float)


def summarise(values: np.ndarray) -> dict[str, float]:
    """Mean, standard deviation, and 95% confidence interval of the mean.

    The standard deviation is the run-to-run spread; the interval is the
    uncertainty in the *average* and shrinks as you add seeds.
    """
    v = values[~np.isnan(values)]
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    half = 1.96 * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half, "n": len(v)}


def sweep(
    cfg: SimConfig,
    name: str,
    values: Iterable[float],
    metric: Metric,
    seeds: Iterable[int] = range(15),
) -> pd.DataFrame:
    """One-at-a-time sensitivity: vary one parameter, repeat over seeds.

    Parameters
    ----------
    cfg : SimConfig
        Base configuration.
    name : str
        Parameter to vary, see ``with_param``.
    values : iterable of float
        Values to try.
    metric : callable
        Output to track.
    seeds : iterable of int
        Seeds to average over at each value.

    Returns
    -------
    pd.DataFrame
        One row per value with columns ``value, mean, sd, lo, hi, n``.
    """
    seeds = list(seeds)
    rows = []
    for v in values:
        stats = summarise(run_over_seeds(with_param(cfg, name, v), metric, seeds))
        rows.append({"value": v, **stats})
    return pd.DataFrame(rows)
