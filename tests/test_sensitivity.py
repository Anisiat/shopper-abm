import numpy as np
import pytest

from abm.config import SimConfig
from abm.sensitivity import (
    kefir_repeat_rate,
    run_over_seeds,
    summarise,
    sweep,
    with_param,
)


def test_with_param_changes_one_beta_and_keeps_original() -> None:
    cfg = SimConfig()
    new = with_param(cfg, "beta.habit", 7.0)
    assert new.beta["habit"] == 7.0
    assert cfg.beta["habit"] == 2.5                       # original untouched
    assert new.beta["price"] == cfg.beta["price"]         # other betas unchanged


def test_with_param_plain_and_invalid() -> None:
    assert with_param(SimConfig(), "habit_rate", 0.5).habit_rate == 0.5
    with pytest.raises(ValueError):
        with_param(SimConfig(), "beta.nonsense", 1.0)
    with pytest.raises(ValueError):
        with_param(SimConfig(), "nonsense", 1.0)
    with pytest.raises(ValueError):                       # config validation still runs
        with_param(SimConfig(), "category_prob", 2.0)


def test_summarise_hand_computed() -> None:
    s = summarise(np.array([1.0, 2.0, 3.0, np.nan]))
    assert s["mean"] == pytest.approx(2.0)
    assert s["sd"] == pytest.approx(1.0)
    assert s["n"] == 3
    assert s["hi"] - s["lo"] == pytest.approx(2 * 1.96 / np.sqrt(3))


def test_run_over_seeds_is_reproducible_and_varies_by_seed() -> None:
    cfg = SimConfig(n_agents=200, n_weeks=14)
    a = run_over_seeds(cfg, kefir_repeat_rate, [0, 1, 2])
    b = run_over_seeds(cfg, kefir_repeat_rate, [0, 1, 2])
    assert np.array_equal(a, b)
    assert len(set(a)) > 1


def test_sweep_shape_and_stronger_habit_raises_repeat() -> None:
    cfg = SimConfig(n_agents=300, n_weeks=20)
    out = sweep(cfg, "beta.habit", [0.0, 12.0], kefir_repeat_rate, seeds=range(6))
    assert list(out["value"]) == [0.0, 12.0]
    assert {"mean", "sd", "lo", "hi", "n"} <= set(out.columns)
    assert out.loc[1, "mean"] > out.loc[0, "mean"]
