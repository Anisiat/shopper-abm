import numpy as np
import pytest

from abm.config import SimConfig
from abm.metrics import compare_scenarios, market_share, repeat_metrics, trial_rate
from abm.products import ATTRIBUTES, default_catalogue, attribute_matrix
from abm.scenarios import (
    BASELINE,
    HealthTrend,
    PriceChange,
    Scenario,
    apply_price_changes,
    health_multiplier,
)
from abm.simulate import SimResult, run_simulation

KEFIR = "Live Kefir (NEW)"
NOTHING = 6


def make_result(choices: np.ndarray) -> SimResult:
    """Wrap a hand-written choice array so metrics can be tested on it."""
    return SimResult(choices, np.zeros_like(choices), np.zeros((choices.shape[1], 6)),
                     None, default_catalogue())  # type: ignore[arg-type]


# ---------- scenarios ----------

def test_price_change_applies_only_in_window_and_to_one_product() -> None:
    products = default_catalogue()
    price = ATTRIBUTES.index("price")
    sc = Scenario("s", price_changes=(PriceChange("Value Fruit Pot", 3, 5, 0.5),))
    X, _ = attribute_matrix(products, 4)

    before = apply_price_changes(sc, products, 2, X)
    during = apply_price_changes(sc, products, 4, X)
    after = apply_price_changes(sc, products, 6, X)

    assert np.array_equal(before, X) and np.array_equal(after, X)
    assert during[0, price] == pytest.approx(0.35)       # 0.70 * 0.5
    assert np.array_equal(during[1:], X[1:])             # other products untouched


def test_apply_price_changes_does_not_modify_input() -> None:
    products = default_catalogue()
    X, _ = attribute_matrix(products, 4)
    X_before = X.copy()
    sc = Scenario("s", price_changes=(PriceChange("Value Fruit Pot", 0, None, 2.0),))
    apply_price_changes(sc, products, 4, X)
    assert np.array_equal(X, X_before)


def test_unknown_product_raises() -> None:
    sc = Scenario("s", price_changes=(PriceChange("Nope", 0, None, 2.0),))
    X, _ = attribute_matrix(default_catalogue(), 0)
    with pytest.raises(ValueError):
        apply_price_changes(sc, default_catalogue(), 0, X)


def test_health_multiplier_switches_on_at_start_week() -> None:
    sc = Scenario("s", health_trend=HealthTrend(10, 1.5))
    assert health_multiplier(sc, 9) == 1.0
    assert health_multiplier(sc, 10) == 1.5
    assert health_multiplier(BASELINE, 20) == 1.0


def test_scenarios_match_baseline_until_they_start() -> None:
    # Common random numbers: same seed => identical draws, so runs only
    # diverge once the scenario starts.
    cfg = SimConfig(n_agents=200, n_weeks=14)
    base = run_simulation(cfg)
    sc = Scenario("s", price_changes=(PriceChange("Classic Brand Fruit", 10, None, 1.5),))
    alt = run_simulation(cfg, sc)
    assert np.array_equal(base.choices[:10], alt.choices[:10])
    assert not np.array_equal(base.choices[10:], alt.choices[10:])


def test_promotion_raises_kefir_share_during_promo() -> None:
    cfg = SimConfig(n_weeks=14)
    sc = Scenario("promo", price_changes=(PriceChange(KEFIR, 8, 11, 0.6),))
    base = market_share(run_simulation(cfg))[KEFIR]
    promo = market_share(run_simulation(cfg, sc))[KEFIR]
    assert promo[8:12].mean() > base[8:12].mean()


def test_price_rise_loses_sales_to_other_products() -> None:
    cfg = SimConfig(n_weeks=20)
    sc = Scenario("rise", price_changes=(PriceChange("Classic Brand Fruit", 10, None, 1.5),))
    brand = "Classic Brand Fruit"
    base = market_share(run_simulation(cfg))[brand]
    rise = market_share(run_simulation(cfg, sc))[brand]
    assert rise[10:].mean() < base[10:].mean()


# ---------- category gate ----------

def test_category_prob_zero_means_nobody_buys() -> None:
    res = run_simulation(SimConfig(n_agents=50, n_weeks=5, category_prob=0.0))
    assert np.all(res.choices == NOTHING)


def test_invalid_category_prob_rejected() -> None:
    with pytest.raises(ValueError):
        SimConfig(category_prob=1.5)


# ---------- metrics ----------

def test_market_share_rows_sum_to_one() -> None:
    res = run_simulation(SimConfig(n_agents=200, n_weeks=8))
    share = market_share(res)
    assert np.allclose(share.sum(axis=1), 1.0)
    assert share[KEFIR][:4].sum() == 0


def test_trial_rate_is_cumulative_and_zero_before_launch() -> None:
    res = run_simulation(SimConfig(n_agents=200, n_weeks=10))
    trial = trial_rate(res, KEFIR)
    assert np.all(np.diff(trial.to_numpy()) >= 0)
    assert trial[:4].sum() == 0


def test_repeat_metrics_hand_computed() -> None:
    # 10 weeks, 4 agents. Kefir = index 5.
    ch = np.full((10, 4), NOTHING)
    ch[4, 0] = 5; ch[6, 0] = 5        # agent 0: first wk 4, repeat 2 weeks later
    ch[5, 1] = 5                      # agent 1: tried, never repeated
    ch[8, 2] = 5                      # agent 2: tried too late to observe (8 + 3 > 9)
    # agent 3 never tried
    m = repeat_metrics(make_result(ch), KEFIR, window=3)

    assert m["trial_rate"] == pytest.approx(3 / 4)
    assert m["n_observable"] == 2                      # agents 0 and 1
    assert m["repeat_rate"] == pytest.approx(1 / 2)
    assert m["median_weeks_to_repeat"] == 2


def test_repeat_metrics_without_buyers_gives_nan() -> None:
    m = repeat_metrics(make_result(np.full((10, 3), NOTHING)), KEFIR)
    assert m["trial_rate"] == 0 and np.isnan(m["repeat_rate"])


def test_compare_scenarios_difference_in_points() -> None:
    cfg = SimConfig(n_agents=200, n_weeks=8)
    sc = Scenario("promo", price_changes=(PriceChange(KEFIR, 4, None, 0.5),))
    out = compare_scenarios({"baseline": run_simulation(cfg), "promo": run_simulation(cfg, sc)})
    assert np.isclose(out["baseline"].sum(), 1.0)
    assert np.isclose(out["promo - baseline"].sum(), 0.0)     # shares shift, total fixed
    assert out.loc[KEFIR, "promo - baseline"] > 0
