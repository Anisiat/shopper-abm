"""Plots for the demo notebook (matplotlib, light surface).

Colour follows the entity: each product keeps the same colour in every chart.
Categorical colours are slots 1-6 of the reference palette, in fixed order.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from abm.metrics import market_share, trial_rate
from abm.sensitivity import KEFIR
from abm.simulate import SimResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BASELINE_COLOUR = INK_2


def product_colours(res: SimResult) -> dict[str, str]:
    """Map each product to a fixed colour (same in every chart).

    The newly launched product takes the lead colour so it stands out; the
    rest follow in catalogue order.
    """
    ordered = sorted(res.products, key=lambda p: p.launch_week == 0)
    return {p.name: SERIES[i] for i, p in enumerate(ordered)}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _fig(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def plot_market_share(res: SimResult, launch_week: int = 4) -> Figure:
    """Weekly market share of every product, with the kefir launch marked."""
    share = market_share(res)
    colours = product_colours(res)
    fig, ax = _fig(figsize=(8, 4.2))
    for name in share.columns:
        ax.plot(share.index, share[name], color=colours[name], lw=2,
                solid_capstyle="round", solid_joinstyle="round", label=name)
    avg = share[KEFIR].iloc[launch_week:].mean()
    last = share[KEFIR].iloc[-1]
    ax.plot(share.index[-1], last, "o", ms=8, color=colours[KEFIR],
            mec=SURFACE, mew=2, zorder=5)
    ax.annotate(f"Kefir: {avg:.0%} avg\nsince launch", (share.index[-1], last),
                xytext=(8, 0), textcoords="offset points", va="center",
                color=INK, fontsize=9)
    ax.axvline(launch_week, color=MUTED, lw=1)
    ax.text(launch_week + 0.2, ax.get_ylim()[1], "kefir launches", color=INK_2,
            fontsize=8.5, va="top")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Week")
    ax.set_xlim(0, share.index[-1] + 5)
    ax.set_title("Market share of purchases, baseline", loc="left", color=INK,
                 fontsize=12)
    ax.legend(ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_2,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return fig


def plot_trial_by_scenario(results: dict[str, SimResult], promo_weeks: tuple[int, int]) -> Figure:
    """Cumulative share of shoppers who have tried the kefir, per scenario."""
    colours = {"baseline": BASELINE_COLOUR, "kefir_promo": SERIES[0],
               "brand_price_rise": SERIES[1], "health_trend": SERIES[2]}
    fig, ax = _fig(figsize=(8, 4))
    for name, res in results.items():
        t = trial_rate(res, KEFIR)
        t = t[t.index >= 4]
        ax.plot(t.index, t.values, lw=2, color=colours.get(name, SERIES[3]),
                solid_capstyle="round", label=name)
    ax.axvspan(promo_weeks[0], promo_weeks[1] + 1, color=MUTED, alpha=0.12, lw=0)
    ax.text(promo_weeks[0] + 0.1, 0.02, "kefir promo", color=INK_2, fontsize=8.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Week")
    ax.set_ylim(0, None)
    ax.set_title("Share of shoppers who have tried the kefir", loc="left",
                 color=INK, fontsize=12)
    ax.legend(ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return fig


def plot_share_change(comparison: pd.DataFrame, res: SimResult) -> Figure:
    """Change in overall share versus baseline, one panel per scenario."""
    cols = [c for c in comparison.columns if c.endswith("- baseline")]
    colours = product_colours(res)
    fig, axes = _fig(1, len(cols), figsize=(3.6 * len(cols), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    limit = max(1.0, comparison[cols].abs().to_numpy().max()) * 1.35
    for ax, col in zip(axes, cols):
        vals = comparison[col]
        ypos = np.arange(len(vals))
        ax.barh(ypos, vals.values, height=0.6,
                color=[colours[n] for n in vals.index])
        ax.axvline(0, color=INK_2, lw=1)
        for y, v in zip(ypos, vals.values):
            ax.annotate(f"{v:+.1f}", (v, y), xytext=(4 if v >= 0 else -4, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right", color=INK, fontsize=8.5)
        ax.set_yticks(ypos, vals.index)
        ax.invert_yaxis()
        ax.set_xlim(-limit, limit)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_title(col.replace(" - baseline", ""), loc="left", color=INK, fontsize=10)
        ax.set_xlabel("points of share")
    fig.suptitle("Change in overall market share vs baseline", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def plot_sensitivity(panels: list[tuple[str, str, pd.DataFrame, float, bool]]) -> Figure:
    """Line-plus-band panels for one-at-a-time sensitivity sweeps.

    Parameters
    ----------
    panels : list of tuple
        Each is ``(title, x label, sweep DataFrame, default x value, is_percent)``.
        The DataFrame comes from ``abm.sensitivity.sweep``; the band is the
        95% confidence interval of the mean across seeds.
    """
    fig, axes = _fig(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (title, xlabel, df, default, pct) in zip(axes, panels):
        ax.fill_between(df["value"], df["lo"], df["hi"], color=SERIES[0], alpha=0.15, lw=0)
        ax.plot(df["value"], df["mean"], color=SERIES[0], lw=2, solid_capstyle="round")
        ax.plot(df["value"], df["mean"], "o", ms=6, color=SERIES[0], mec=SURFACE, mew=2)
        ax.axvline(default, color=MUTED, lw=1)
        ax.text(default, ax.get_ylim()[1], " default", color=INK_2, fontsize=8.5, va="top")
        if pct:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    return fig
