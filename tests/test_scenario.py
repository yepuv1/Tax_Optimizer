"""Tests for `tax_optimizer.scenario` (JSON loader + --set parser + serializer)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tax_optimizer import (
    Config,
    Inputs,
    Mortality,
    ScenarioError,
    apply_scenario,
    apply_set_overrides,
    load_scenario_file,
    scenario_to_dict,
)
from tax_optimizer.market import (
    AssetLocation,
    BootstrapModel,
    DeterministicModel,
    LognormalModel,
)
from tax_optimizer.spending import LongTermCareShock, SpendingProfile
from tax_optimizer.tax.regimes import (
    PRE_TCJA_2017,
    SUNSET_2026,
    TCJA_EXTENDED,
)


# ---------------------------------------------------------------------------
# load_scenario_file
# ---------------------------------------------------------------------------


class TestLoadScenarioFile:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ScenarioError, match="not found"):
            load_scenario_file(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        with pytest.raises(ScenarioError, match="Invalid JSON"):
            load_scenario_file(p)

    def test_non_object_root_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]")
        with pytest.raises(ScenarioError, match="must be a JSON object"):
            load_scenario_file(p)

    def test_round_trip_with_path_or_str(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.json"
        p.write_text(json.dumps({"config": {"horizon_age": 95}}))
        from_path = load_scenario_file(p)
        from_str = load_scenario_file(str(p))
        assert from_path == from_str == {"config": {"horizon_age": 95}}


# ---------------------------------------------------------------------------
# apply_scenario: top-level shape + unknown-key errors
# ---------------------------------------------------------------------------


class TestApplyScenario:
    def test_empty_scenario_is_no_op(self) -> None:
        cfg, inp = apply_scenario(Config(), Inputs(), {})
        assert cfg == Config()
        assert inp == Inputs()

    def test_unknown_top_level_key_raises(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown top-level"):
            apply_scenario(Config(), Inputs(), {"settings": {}})

    def test_unknown_config_key_raises(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown config field"):
            apply_scenario(Config(), Inputs(), {"config": {"made_up": 1}})

    def test_legacy_spouse_under_config_raises_with_hint(self) -> None:
        # The user is on an old scenario file that still puts spouse_*
        # under config. The error should point at the new home (inputs).
        with pytest.raises(ScenarioError, match=r"moved to `inputs`"):
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"spouse_a_age_start": 52}},
            )

    def test_legacy_ss_start_age_under_config_raises_with_hint(self) -> None:
        # SS claim age moved from cfg.ss_start_age to inputs.ss.start_age.
        with pytest.raises(
            ScenarioError, match=r"ss_start_age -> inputs\.ss\.start_age"
        ):
            apply_scenario(
                Config(), Inputs(), {"config": {"ss_start_age": 70}}
            )

    def test_legacy_pension_start_age_under_config_raises_with_hint(self) -> None:
        with pytest.raises(
            ScenarioError,
            match=r"pension_start_age -> inputs\.pension\.start_age",
        ):
            apply_scenario(
                Config(), Inputs(), {"config": {"pension_start_age": 65}}
            )

    def test_new_ss_start_age_location_applies(self) -> None:
        cfg, inp = apply_scenario(
            Config(),
            Inputs(),
            {"inputs": {"ss": {"start_age": 67}}},
        )
        assert inp.ss.start_age == 67
        # Other ss fields keep defaults.
        assert inp.ss.monthly_spouse_a == Inputs().ss.monthly_spouse_a

    def test_new_pension_start_age_location_applies(self) -> None:
        cfg, inp = apply_scenario(
            Config(),
            Inputs(),
            {"inputs": {"pension": {"start_age": 67, "monthly_at_nrd": 1500}}},
        )
        assert inp.pension.start_age == 67
        assert inp.pension.monthly_at_nrd == 1500

    def test_unknown_inputs_key_raises(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown inputs field"):
            apply_scenario(Config(), Inputs(), {"inputs": {"bogus": 1}})

    def test_unknown_nested_inputs_subkey_raises(self) -> None:
        with pytest.raises(ScenarioError, match=r"Unknown inputs.starting field"):
            apply_scenario(
                Config(), Inputs(), {"inputs": {"starting": {"bogus": 1}}}
            )

    def test_simple_scalar_overrides_apply(self) -> None:
        # `annual_expenses` is deprecated and emits a DeprecationWarning;
        # the value is still round-tripped through the loader for legacy
        # scenario files. See `Inputs.__post_init__`.
        with pytest.warns(DeprecationWarning, match="annual_expenses"):
            cfg, inp = apply_scenario(
                Config(),
                Inputs(),
                {
                    "config": {"horizon_age": 95, "inflation": 0.03},
                    "inputs": {
                        "spouse_a_age_start": 52,
                        "annual_expenses": 95_000,
                    },
                },
            )
        assert cfg.horizon_age == 95
        assert cfg.inflation == 0.03
        assert inp.spouse_a_age_start == 52
        assert inp.annual_expenses == 95_000


# ---------------------------------------------------------------------------
# Coercion: tax_regime / market / spending / mortality / asset_location
# ---------------------------------------------------------------------------


class TestRegimeCoercion:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("tcja", TCJA_EXTENDED),
            ("tcja_extended", TCJA_EXTENDED),
            ("pre_tcja", PRE_TCJA_2017),
            ("pre_tcja_2017", PRE_TCJA_2017),
            ("sunset", SUNSET_2026),
            ("sunset_2026", SUNSET_2026),
            ("SUNSET", SUNSET_2026),  # case-insensitive
            ("  tcja  ", TCJA_EXTENDED),  # whitespace tolerant
        ],
    )
    def test_known_regime_names(self, name: str, expected) -> None:
        cfg, _ = apply_scenario(
            Config(), Inputs(), {"config": {"tax_regime": name}}
        )
        assert cfg.tax_regime is expected

    def test_unknown_regime_raises(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown tax_regime"):
            apply_scenario(
                Config(), Inputs(), {"config": {"tax_regime": "bogus_2030"}}
            )

    def test_non_string_regime_raises(self) -> None:
        with pytest.raises(ScenarioError, match="must be a string"):
            apply_scenario(Config(), Inputs(), {"config": {"tax_regime": 5}})

    def test_regime_change_target_can_be_null(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {"config": {"regime_change_target": None}},
        )
        assert cfg.regime_change_target is None


class TestMarketCoercion:
    def test_deterministic_default(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {"config": {"market": {"kind": "deterministic", "equity": 0.05, "bond": 0.03}}},
        )
        assert isinstance(cfg.market, DeterministicModel)
        assert cfg.market.equity == 0.05
        assert cfg.market.bond == 0.03

    def test_lognormal(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {
                "config": {
                    "market": {
                        "kind": "lognormal",
                        "equity_mu": 0.07,
                        "equity_sigma": 0.18,
                        "bond_mu": 0.04,
                        "bond_sigma": 0.06,
                    }
                }
            },
        )
        assert isinstance(cfg.market, LognormalModel)
        assert cfg.market.equity_mu == 0.07

    def test_bootstrap_with_explicit_history(self) -> None:
        # History must be at least as long as the block size — see the
        # F9 validator added in v6.2 (`block_size > len(history)`
        # makes `rng.integers(0, n_hist - block_size + 1)` reject a
        # non-positive `high`, breaking `begin_path`).
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {
                "config": {
                    "market": {
                        "kind": "bootstrap",
                        "block_size": 3,
                        "equity_history": [0.1, 0.2, -0.05, 0.08, 0.04],
                        "bond_history": [0.04, 0.05, 0.03, 0.02, 0.06],
                    }
                }
            },
        )
        assert isinstance(cfg.market, BootstrapModel)
        assert cfg.market.block_size == 3
        # Tuples preferred internally for hashability/immutability.
        assert isinstance(cfg.market.equity_history, tuple)

    def test_unknown_market_kind_raises(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown market kind"):
            apply_scenario(
                Config(), Inputs(), {"config": {"market": {"kind": "wat"}}}
            )

    def test_unknown_market_param_raises(self) -> None:
        with pytest.raises(ScenarioError, match=r"Unknown market\(lognormal\) key"):
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"market": {"kind": "lognormal", "wibble": 1}}},
            )


class TestSpendingCoercion:
    def test_flat(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {
                "config": {
                    "spending": {
                        "kind": "flat",
                        "base_spending": 90_000,
                        "inflation": 0.03,
                    }
                }
            },
        )
        assert isinstance(cfg.spending, SpendingProfile)
        assert cfg.spending.base_spending == 90_000
        assert cfg.spending.inflation == 0.03
        assert cfg.spending.ltc_shock is None

    def test_smile(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {
                "config": {
                    "spending": {
                        "kind": "smile",
                        "base_spending": 100_000,
                        "inflation": 0.025,
                        "ltc_years": 4,
                        "ltc_annual_today": 100_000,
                    }
                }
            },
        )
        assert isinstance(cfg.spending.ltc_shock, LongTermCareShock)
        assert cfg.spending.ltc_shock.years == 4
        assert cfg.spending.ltc_shock.annual_cost_today == 100_000

    def test_unknown_spending_kind_raises(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown spending kind"):
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"spending": {"kind": "wat", "base_spending": 1}}},
            )


class TestMortalityCoercion:
    def test_partial_dict_replaces_only_specified_fields(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {"config": {"mortality": {"year_of_death_a": 25}}},
        )
        assert isinstance(cfg.mortality, Mortality)
        assert cfg.mortality.year_of_death_a == 25
        # Unspecified fields keep their defaults.
        assert cfg.mortality.pension_survivor_pct == 0.50
        assert cfg.mortality.year_of_death_b is None

    def test_unknown_mortality_field_raises(self) -> None:
        with pytest.raises(ScenarioError, match="Unknown mortality"):
            apply_scenario(
                Config(),
                Inputs(),
                {"config": {"mortality": {"bogus_year": 5}}},
            )


class TestAssetLocationCoercion:
    def test_uniform_shortcut(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {"config": {"asset_location": {"uniform_equity_pct": 0.7}}},
        )
        assert isinstance(cfg.asset_location, AssetLocation)
        for bucket in (
            cfg.asset_location.pretax,
            cfg.asset_location.roth,
            cfg.asset_location.taxable,
            cfg.asset_location.hsa,
        ):
            assert bucket.equity_pct == 0.7

    def test_uniform_with_other_keys_rejected(self) -> None:
        with pytest.raises(ScenarioError, match="cannot be combined"):
            apply_scenario(
                Config(),
                Inputs(),
                {
                    "config": {
                        "asset_location": {
                            "uniform_equity_pct": 0.7,
                            "roth_equity_pct": 1.0,
                        }
                    }
                },
            )

    def test_per_bucket_overrides(self) -> None:
        cfg, _ = apply_scenario(
            Config(),
            Inputs(),
            {
                "config": {
                    "asset_location": {
                        "pretax_equity_pct": 0.4,
                        "roth_equity_pct": 1.0,
                        "taxable_equity_pct": 0.8,
                        "hsa_equity_pct": 0.9,
                    }
                }
            },
        )
        assert cfg.asset_location.pretax.equity_pct == 0.4
        assert cfg.asset_location.roth.equity_pct == 1.0
        assert cfg.asset_location.taxable.equity_pct == 0.8
        assert cfg.asset_location.hsa.equity_pct == 0.9


# ---------------------------------------------------------------------------
# --set DOTTED.PATH=VALUE parsing
# ---------------------------------------------------------------------------


class TestApplySetOverrides:
    def test_no_overrides_is_no_op(self) -> None:
        cfg, inp = apply_set_overrides(Config(), Inputs(), [])
        assert cfg == Config()
        assert inp == Inputs()

    def test_simple_int_override(self) -> None:
        cfg, inp = apply_set_overrides(
            Config(), Inputs(), ["config.horizon_age=95"]
        )
        assert cfg.horizon_age == 95

    def test_simple_float_override(self) -> None:
        # Deprecated knob; --set still threads the value through but
        # the simulator ignores it. The deprecation warning is part of
        # the asserted behavior so we know it fires on every legacy use.
        with pytest.warns(DeprecationWarning, match="annual_expenses"):
            cfg, inp = apply_set_overrides(
                Config(), Inputs(), ["inputs.annual_expenses=120000"]
            )
        assert inp.annual_expenses == 120_000

    def test_string_value_falls_back_to_bare_token(self) -> None:
        # `sunset` is not valid JSON but the loader falls back to a string.
        cfg, _ = apply_set_overrides(
            Config(), Inputs(), ["config.tax_regime=sunset"]
        )
        assert cfg.tax_regime is SUNSET_2026

    def test_quoted_string_value(self) -> None:
        cfg, _ = apply_set_overrides(
            Config(), Inputs(), ['config.tax_regime="tcja"']
        )
        assert cfg.tax_regime is TCJA_EXTENDED

    def test_dotted_path_into_nested_dataclass(self) -> None:
        cfg, inp = apply_set_overrides(
            Config(), Inputs(), ["inputs.starting.hsa=25000"]
        )
        assert inp.starting.hsa == 25_000

    def test_json_literal_value(self) -> None:
        cfg, _ = apply_set_overrides(
            Config(),
            Inputs(),
            [
                'config.market={"kind":"lognormal","equity_mu":0.07,'
                '"equity_sigma":0.18,"bond_mu":0.04,"bond_sigma":0.06}'
            ],
        )
        assert isinstance(cfg.market, LognormalModel)
        assert cfg.market.equity_mu == 0.07

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(ScenarioError, match="missing '='"):
            apply_set_overrides(Config(), Inputs(), ["config.horizon_age"])

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ScenarioError, match="empty key"):
            apply_set_overrides(Config(), Inputs(), ["=5"])

    def test_bad_root_raises(self) -> None:
        with pytest.raises(ScenarioError, match="must start with"):
            apply_set_overrides(Config(), Inputs(), ["other.foo=1"])

    def test_root_only_raises(self) -> None:
        with pytest.raises(ScenarioError, match="missing a field"):
            apply_set_overrides(Config(), Inputs(), ["config="])

    def test_legacy_config_spouse_field_routes_with_hint(self) -> None:
        with pytest.raises(ScenarioError, match=r"use inputs\.spouse_a_age_start"):
            apply_set_overrides(
                Config(),
                Inputs(),
                ["config.spouse_a_age_start=52"],
            )

    def test_legacy_config_ss_start_age_routes_with_hint(self) -> None:
        with pytest.raises(ScenarioError, match=r"use inputs\.ss\.start_age"):
            apply_set_overrides(
                Config(), Inputs(), ["config.ss_start_age=68"]
            )

    def test_legacy_config_pension_start_age_routes_with_hint(self) -> None:
        with pytest.raises(
            ScenarioError, match=r"use inputs\.pension\.start_age"
        ):
            apply_set_overrides(
                Config(), Inputs(), ["config.pension_start_age=67"]
            )

    def test_new_set_path_inputs_ss_start_age(self) -> None:
        _, inp = apply_set_overrides(
            Config(), Inputs(), ["inputs.ss.start_age=67"]
        )
        assert inp.ss.start_age == 67

    def test_new_set_path_inputs_pension_start_age(self) -> None:
        _, inp = apply_set_overrides(
            Config(), Inputs(), ["inputs.pension.start_age=67"]
        )
        assert inp.pension.start_age == 67

    def test_multiple_overrides_compose(self) -> None:
        cfg, inp = apply_set_overrides(
            Config(),
            Inputs(),
            [
                "config.horizon_age=95",
                "inputs.starting.hsa=25000",
                "inputs.spouse_a_age_start=52",
            ],
        )
        assert cfg.horizon_age == 95
        assert inp.starting.hsa == 25_000
        assert inp.spouse_a_age_start == 52


# ---------------------------------------------------------------------------
# scenario_to_dict (round-trip property)
# ---------------------------------------------------------------------------


class TestScenarioToDict:
    def test_round_trip_defaults(self) -> None:
        # Defaults must round-trip exactly (modulo regime → name string).
        cfg0, inp0 = Config(), Inputs()
        d = scenario_to_dict(cfg0, inp0)
        assert "config" in d
        assert "inputs" in d
        cfg1, inp1 = apply_scenario(Config(), Inputs(), d)
        assert cfg1.horizon_age == cfg0.horizon_age
        assert cfg1.tax_regime is cfg0.tax_regime
        assert inp1.spouse_a_age_start == inp0.spouse_a_age_start
        assert inp1.starting.hsa == inp0.starting.hsa

    def test_round_trip_with_market_lognormal(self) -> None:
        cfg0 = Config(
            market=LognormalModel(
                equity_mu=0.075, equity_sigma=0.19, bond_mu=0.04, bond_sigma=0.06
            )
        )
        inp0 = Inputs()
        d = scenario_to_dict(cfg0, inp0)
        assert d["config"]["market"]["kind"] == "lognormal"
        cfg1, _ = apply_scenario(Config(), Inputs(), d)
        assert isinstance(cfg1.market, LognormalModel)
        assert cfg1.market.equity_mu == 0.075
        assert cfg1.market.equity_sigma == 0.19

    def test_round_trip_with_set_overrides(self) -> None:
        cfg0, inp0 = apply_set_overrides(
            Config(),
            Inputs(),
            [
                "config.horizon_age=95",
                "config.tax_regime=sunset",
                "inputs.starting.hsa=25000",
            ],
        )
        d = scenario_to_dict(cfg0, inp0)
        cfg1, inp1 = apply_scenario(Config(), Inputs(), d)
        assert cfg1.horizon_age == 95
        assert cfg1.tax_regime is SUNSET_2026
        assert inp1.starting.hsa == 25_000

    def test_serialized_regime_is_string_name(self) -> None:
        d = scenario_to_dict(Config(), Inputs())
        # tax_regime is encoded as a short name, not a dataclass blob
        assert isinstance(d["config"]["tax_regime"], str)
