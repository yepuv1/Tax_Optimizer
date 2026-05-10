"""Tests for the new market-model pieces:

* Equity-bond correlation in `LognormalModel`
* CAPE-conditioning on `LognormalModel.equity_mu`
* `HistoricalSequenceModel`
* `CMA_PRESETS` + `lognormal_from_cma`
* Scenario JSON loader/serializer round-trips for all of the above
"""

from __future__ import annotations

import numpy as np
import pytest

from tax_optimizer import (
    CMA_PRESETS,
    BootstrapModel,
    Config,
    HistoricalSequenceModel,
    Inputs,
    LognormalModel,
    apply_scenario,
    lognormal_from_cma,
    scenario_to_dict,
    simulate,
)
from tax_optimizer.market import _HIST_BOND, _HIST_EQUITY


# ---------------------------------------------------------------------------
# 1. Equity-bond correlation
# ---------------------------------------------------------------------------


class TestEquityBondCorrelation:
    def test_default_correlation_is_positive(self) -> None:
        m = LognormalModel()
        assert m.equity_bond_corr == pytest.approx(0.10)

    def test_correlation_validation(self) -> None:
        with pytest.raises(ValueError):
            LognormalModel(equity_bond_corr=1.5)
        with pytest.raises(ValueError):
            LognormalModel(equity_bond_corr=-2.0)

    def test_zero_correlation_recovers_independent_draws(self) -> None:
        rng = np.random.default_rng(0)
        m = LognormalModel(
            equity_mu=0.07,
            equity_sigma=0.18,
            bond_mu=0.04,
            bond_sigma=0.06,
            equity_bond_corr=0.0,
        )
        m.begin_path(50_000, rng)
        eq = m._equity_path
        bd = m._bond_path
        corr = float(np.corrcoef(eq, bd)[0, 1])
        assert abs(corr) < 0.02

    def test_positive_correlation_propagates_to_samples(self) -> None:
        rng = np.random.default_rng(1)
        m = LognormalModel(
            equity_mu=0.07,
            equity_sigma=0.18,
            bond_mu=0.04,
            bond_sigma=0.06,
            equity_bond_corr=0.50,
        )
        m.begin_path(50_000, rng)
        corr = float(np.corrcoef(m._equity_path, m._bond_path)[0, 1])
        assert 0.45 < corr < 0.55

    def test_negative_correlation_propagates_to_samples(self) -> None:
        rng = np.random.default_rng(2)
        m = LognormalModel(equity_bond_corr=-0.40)
        m.begin_path(50_000, rng)
        corr = float(np.corrcoef(m._equity_path, m._bond_path)[0, 1])
        assert -0.45 < corr < -0.35

    def test_marginals_unchanged_by_correlation(self) -> None:
        """Correlation rotates joint draws; the per-asset means/sigmas
        should still match what we requested."""
        rng = np.random.default_rng(3)
        m = LognormalModel(
            equity_mu=0.08,
            equity_sigma=0.20,
            bond_mu=0.04,
            bond_sigma=0.07,
            equity_bond_corr=0.30,
        )
        m.begin_path(100_000, rng)
        assert abs(float(m._equity_path.mean()) - 0.08) < 0.005
        assert abs(float(m._equity_path.std()) - 0.20) < 0.005
        assert abs(float(m._bond_path.mean()) - 0.04) < 0.005
        assert abs(float(m._bond_path.std()) - 0.07) < 0.005


# ---------------------------------------------------------------------------
# 2. CAPE conditioning
# ---------------------------------------------------------------------------


class TestCAPEConditioning:
    def test_unset_cape_uses_raw_mu(self) -> None:
        m = LognormalModel(equity_mu=0.08)
        assert m.effective_equity_mu() == pytest.approx(0.08)

    def test_high_cape_lowers_expected_equity_return(self) -> None:
        # CAPE 33 vs long-run 16.5 → halve mu.
        m = LognormalModel(equity_mu=0.08, cape_today=33.0, cape_long_run=16.5)
        assert m.effective_equity_mu() == pytest.approx(0.04)

    def test_low_cape_raises_expected_equity_return(self) -> None:
        # CAPE 11 vs long-run 16.5 → boost mu by 50%.
        m = LognormalModel(equity_mu=0.07, cape_today=11.0, cape_long_run=16.5)
        assert m.effective_equity_mu() == pytest.approx(0.07 * 16.5 / 11.0)

    def test_cape_at_long_run_is_no_op(self) -> None:
        m = LognormalModel(equity_mu=0.07, cape_today=16.5, cape_long_run=16.5)
        assert m.effective_equity_mu() == pytest.approx(0.07)

    def test_cape_zero_or_none_is_no_op(self) -> None:
        assert LognormalModel(equity_mu=0.07, cape_today=None).effective_equity_mu() \
            == pytest.approx(0.07)
        assert LognormalModel(equity_mu=0.07, cape_today=0.0).effective_equity_mu() \
            == pytest.approx(0.07)

    def test_cape_actually_lowers_simulated_returns(self) -> None:
        rng = np.random.default_rng(4)
        m_high = LognormalModel(equity_mu=0.08, cape_today=33.0)
        m_high.begin_path(20_000, rng)
        rng = np.random.default_rng(4)
        m_off = LognormalModel(equity_mu=0.08, cape_today=None)
        m_off.begin_path(20_000, rng)

        assert float(m_high._equity_path.mean()) < float(m_off._equity_path.mean()) - 0.01
        # Bond mu should be unaffected.
        assert abs(float(m_high._bond_path.mean()) - float(m_off._bond_path.mean())) < 0.005

    def test_cape_does_not_change_volatility(self) -> None:
        rng = np.random.default_rng(5)
        m = LognormalModel(equity_mu=0.08, equity_sigma=0.18, cape_today=33.0)
        m.begin_path(50_000, rng)
        # σ should still be ~0.18 even though μ was scaled.
        assert abs(float(m._equity_path.std()) - 0.18) < 0.005


# ---------------------------------------------------------------------------
# 3. HistoricalSequenceModel
# ---------------------------------------------------------------------------


class TestHistoricalSequenceModel:
    def test_returns_match_history_at_chosen_window(self) -> None:
        m = HistoricalSequenceModel()
        rng = np.random.default_rng(6)
        m.begin_path(30, rng)
        # Path equals an actual contiguous slice of history.
        path_eq = tuple(round(x, 12) for x in m._equity_path.tolist())
        rolled = [
            tuple(round(x, 12) for x in _HIST_EQUITY[i : i + 30])
            for i in range(len(_HIST_EQUITY) - 30 + 1)
        ]
        assert path_eq in rolled

    def test_path_is_contiguous(self) -> None:
        """Two consecutive years in the sampled path must come from
        consecutive years in history. (This is the whole point of the
        model — preserve sequence-of-returns shape.)"""
        m = HistoricalSequenceModel()
        rng = np.random.default_rng(7)
        m.begin_path(30, rng)
        # Find the start index by matching the first equity return.
        eq0 = float(m._equity_path[0])
        starts = [
            i
            for i, v in enumerate(_HIST_EQUITY[: len(_HIST_EQUITY) - 30 + 1])
            if abs(v - eq0) < 1e-12
        ]
        assert starts, "first sampled year should appear somewhere in history"
        # At least one candidate start should match the full 30-year slice.
        assert any(
            np.allclose(m._equity_path, _HIST_EQUITY[s : s + 30])
            and np.allclose(m._bond_path, _HIST_BOND[s : s + 30])
            for s in starts
        )

    def test_rejects_horizon_longer_than_history(self) -> None:
        m = HistoricalSequenceModel()
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="exceeds available history"):
            m.begin_path(len(_HIST_EQUITY) + 5, rng)

    def test_full_history_horizon_yields_only_one_sequence(self) -> None:
        m = HistoricalSequenceModel()
        n = len(_HIST_EQUITY)
        for seed in range(5):
            rng = np.random.default_rng(seed)
            m.begin_path(n, rng)
            assert np.allclose(m._equity_path, _HIST_EQUITY)
            assert np.allclose(m._bond_path, _HIST_BOND)

    def test_rejects_mismatched_history_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            HistoricalSequenceModel(
                equity_history=(0.1, 0.2, 0.3),
                bond_history=(0.05, 0.05),
            )

    def test_works_inside_simulator(self) -> None:
        cfg = Config(market=HistoricalSequenceModel())
        df = simulate(cfg, Inputs(), rng=np.random.default_rng(42))
        assert len(df) > 0


# ---------------------------------------------------------------------------
# 4. CMA presets
# ---------------------------------------------------------------------------


class TestCMAPresets:
    def test_known_presets_exist(self) -> None:
        for name in (
            "vanguard_2025",
            "jpm_ltcma_2025",
            "horizon_2025",
            "historical_1928_2023",
            "historical_1985_2023",
        ):
            assert name in CMA_PRESETS

    def test_preset_dict_has_expected_keys(self) -> None:
        required = {
            "equity_mu",
            "equity_sigma",
            "bond_mu",
            "bond_sigma",
            "equity_bond_corr",
        }
        for name, params in CMA_PRESETS.items():
            assert required <= set(params), f"preset {name} missing keys"

    def test_lognormal_from_cma_returns_lognormal_with_right_params(self) -> None:
        m = lognormal_from_cma("vanguard_2025")
        params = CMA_PRESETS["vanguard_2025"]
        assert isinstance(m, LognormalModel)
        assert m.equity_mu == pytest.approx(params["equity_mu"])
        assert m.equity_sigma == pytest.approx(params["equity_sigma"])
        assert m.bond_mu == pytest.approx(params["bond_mu"])
        assert m.bond_sigma == pytest.approx(params["bond_sigma"])
        assert m.equity_bond_corr == pytest.approx(params["equity_bond_corr"])

    def test_lognormal_from_cma_overrides_apply(self) -> None:
        m = lognormal_from_cma("jpm_ltcma_2025", cape_today=33.0, equity_sigma=0.20)
        assert m.cape_today == pytest.approx(33.0)
        assert m.equity_sigma == pytest.approx(0.20)
        # Other params still come from preset.
        assert m.equity_mu == pytest.approx(CMA_PRESETS["jpm_ltcma_2025"]["equity_mu"])

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown CMA preset"):
            lognormal_from_cma("not_a_real_preset")

    def test_vanguard_more_conservative_than_historical(self) -> None:
        v = CMA_PRESETS["vanguard_2025"]
        h = CMA_PRESETS["historical_1928_2023"]
        # The whole point of forward-looking CMAs is to be lower than
        # history; if this assertion fires, somebody copied the wrong
        # numbers.
        assert v["equity_mu"] < h["equity_mu"]


# ---------------------------------------------------------------------------
# 5. Scenario JSON round-trips
# ---------------------------------------------------------------------------


class TestScenarioRoundTrip:
    def test_lognormal_with_correlation_round_trips(self) -> None:
        cfg0 = Config(
            market=LognormalModel(
                equity_mu=0.07,
                equity_sigma=0.18,
                bond_mu=0.04,
                bond_sigma=0.06,
                equity_bond_corr=0.25,
            )
        )
        d = scenario_to_dict(cfg0, Inputs())
        assert d["config"]["market"]["equity_bond_corr"] == pytest.approx(0.25)
        cfg1, _ = apply_scenario(Config(), Inputs(), d)
        assert isinstance(cfg1.market, LognormalModel)
        assert cfg1.market.equity_bond_corr == pytest.approx(0.25)

    def test_lognormal_with_cape_round_trips(self) -> None:
        cfg0 = Config(
            market=LognormalModel(
                equity_mu=0.08,
                cape_today=33.0,
                cape_long_run=16.5,
            )
        )
        d = scenario_to_dict(cfg0, Inputs())
        assert d["config"]["market"]["cape_today"] == pytest.approx(33.0)
        cfg1, _ = apply_scenario(Config(), Inputs(), d)
        assert isinstance(cfg1.market, LognormalModel)
        assert cfg1.market.cape_today == pytest.approx(33.0)
        assert cfg1.market.cape_long_run == pytest.approx(16.5)

    def test_cma_shortcut_in_scenario(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {"config": {"market": {"cma": "vanguard_2025"}}},
        )
        assert isinstance(cfg.market, LognormalModel)
        params = CMA_PRESETS["vanguard_2025"]
        assert cfg.market.equity_mu == pytest.approx(params["equity_mu"])
        assert cfg.market.equity_bond_corr == pytest.approx(params["equity_bond_corr"])

    def test_cma_shortcut_with_overrides(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {
                "config": {
                    "market": {
                        "cma": "vanguard_2025",
                        "cape_today": 30.0,
                        "equity_sigma": 0.22,
                    }
                }
            },
        )
        assert isinstance(cfg.market, LognormalModel)
        assert cfg.market.cape_today == pytest.approx(30.0)
        assert cfg.market.equity_sigma == pytest.approx(0.22)
        # Non-overridden field comes from preset.
        params = CMA_PRESETS["vanguard_2025"]
        assert cfg.market.equity_mu == pytest.approx(params["equity_mu"])

    def test_cma_unknown_preset_raises(self) -> None:
        from tax_optimizer.scenario import ScenarioError

        with pytest.raises(ScenarioError, match="Unknown CMA preset"):
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"market": {"cma": "not_real"}}},
            )

    def test_cma_with_bootstrap_kind_raises(self) -> None:
        from tax_optimizer.scenario import ScenarioError

        with pytest.raises(ScenarioError, match="cma is only valid"):
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"market": {"kind": "bootstrap", "cma": "vanguard_2025"}}},
            )

    def test_historical_sequence_round_trips(self) -> None:
        cfg0 = Config(market=HistoricalSequenceModel())
        d = scenario_to_dict(cfg0, Inputs())
        assert d["config"]["market"]["kind"] == "historical_sequence"
        cfg1, _ = apply_scenario(Config(), Inputs(), d)
        assert isinstance(cfg1.market, HistoricalSequenceModel)

    def test_unknown_market_kind_raises(self) -> None:
        from tax_optimizer.scenario import ScenarioError

        with pytest.raises(ScenarioError, match="Unknown market kind"):
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"market": {"kind": "made_up_model"}}},
            )
