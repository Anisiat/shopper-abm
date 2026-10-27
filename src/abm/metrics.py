"""Outputs: market share, trial, repeat purchase and scenario comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from abm.simulate import SimResult


def _index(res: SimResult, product: str) -> int:
    names = [p.name for p in res.products]
    if product not in names:
        raise ValueError(f"Unknown product {product!r}")
    return names.index(product)


def market_share(res: SimResult) -> pd.DataFrame:
    """Share of each product among purchases, by week.

    Weeks with no purchases at all give NaN. "Bought nothing" is excluded.

    Parameters
    ----------
    res : SimResult
        Output of ``run_simulation``.

    Returns
    -------
    pd.DataFrame
        Rows are weeks, columns are products, each row sums to 1.
    """
    names = [p.name for p in res.products]
    j = len(names)
    counts = np.stack([(res.choices == k).sum(axis=1) for k in range(j)], axis=1)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.full(counts.shape, np.nan), where=totals > 0)
    return pd.DataFrame(share, columns=names).rename_axis("week")


def trial_rate(res: SimResult, product: str) -> pd.Series:
    """Cumulative share of all shoppers who have bought the product at least once.

    Parameters
    ----------
    res : SimResult
        Output of ``run_simulation``.
    product : str
        Product name.

    Returns
    -------
    pd.Series
        Indexed by week; non-decreasing.
    """
    bought = res.choices == _index(res, product)
    ever = np.cumsum(bought, axis=0) > 0
    return pd.Series(ever.mean(axis=1), name=f"trial: {product}").rename_axis("week")


def repeat_metrics(res: SimResult, product: str, window: int = 8) -> dict[str, float]:
    """Repeat purchase rate and time to repurchase for one product.

    A shopper is a *repeater* if they buy the product again within ``window``
    weeks of their first purchase. Only shoppers who tried it early enough to
    be observed for the full window are counted, so late triers (whose
    follow-up is cut off by the end of the run) don't bias the rate down.

    Parameters
    ----------
    res : SimResult
        Output of ``run_simulation``.
    product : str
        Product name.
    window : int
        Weeks after the first purchase in which a repeat counts.

    Returns
    -------
    dict
        ``trial_rate`` (share who ever bought), ``repeat_rate`` (share of
        observable triers who repeated), ``median_weeks_to_repeat`` (among
        repeaters), and ``n_observable`` (triers in the denominator).
    """
    bought = res.choices == _index(res, product)           # (weeks, agents)
    n_weeks, n_agents = bought.shape
    tried = bought.any(axis=0)
    first = bought.argmax(axis=0)                          # first purchase week

    observable = tried & (first + window <= n_weeks - 1)
    cum = np.cumsum(bought, axis=0)
    agent = np.arange(n_agents)
    end = np.minimum(first + window, n_weeks - 1)
    repeats_in_window = cum[end, agent] - cum[first, agent]
    repeater = observable & (repeats_in_window > 0)

    # Weeks from first purchase to the next purchase, for observable repeaters.
    gaps = []
    for a in np.flatnonzero(repeater):
        later = np.flatnonzero(bought[first[a] + 1:, a])
        gaps.append(later[0] + 1)

    n_obs = int(observable.sum())
    return {
        "trial_rate": float(tried.mean()),
        "repeat_rate": float(repeater.sum() / n_obs) if n_obs else float("nan"),
        "median_weeks_to_repeat": float(np.median(gaps)) if gaps else float("nan"),
        "n_observable": n_obs,
    }


def compare_scenarios(
    results: dict[str, SimResult], baseline: str = "baseline"
) -> pd.DataFrame:
    """Overall market share per scenario and its difference from baseline.

    Parameters
    ----------
    results : dict[str, SimResult]
        Scenario name to result. Must contain ``baseline``.
    baseline : str
        Key of the reference scenario.

    Returns
    -------
    pd.DataFrame
        Rows are products; one share column per scenario, plus a
        ``<scenario> - baseline`` column in percentage points.
    """
    shares = {}
    for name, res in results.items():
        counts = np.array(
            [(res.choices == k).sum() for k in range(len(res.products))], dtype=float
        )
        shares[name] = counts / counts.sum()
    names = [p.name for p in results[baseline].products]
    df = pd.DataFrame(shares, index=names)
    for name in results:
        if name != baseline:
            df[f"{name} - {baseline}"] = (df[name] - df[baseline]) * 100
    return df
