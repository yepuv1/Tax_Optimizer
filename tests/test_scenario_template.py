"""Drift tests for `scenarios/template.json`.

`scenarios/template.json` is the canonical "every knob, with its default
value" reference. Users copy it as the starting point for a custom
scenario; the simulator parses it without an `Unknown field` error
because every public dataclass field is present.

These tests fail (loudly) when a new field is added to `Config`,
`Inputs`, or one of the nested Input blocks WITHOUT also being
added to `scenarios/template.json`. That's the whole maintenance
contract: the template stays in sync with the dataclass surface so
the next person to read the file gets an accurate menu.

When you add a new knob:

  1. Add the dataclass field with its default.
  2. Add the same field+default to `scenarios/template.json`.
  3. If the knob is interesting/non-zero, also reflect it in
     `scenarios/example01.json` / `scenarios/example02.json` (the
     two narrative example scenarios).
  4. The tests in this file should turn green again.

The `market` and `spending` blocks are polymorphic (they accept
multiple "kind" tags); the template renders ONE representative
shape (lognormal + smile). Drift tests focus on flat-config /
flat-inputs fields plus the well-known nested dataclasses
(`StartingBalances`, `CurrentIncome`, `CurrentContrib`,
`PensionInputs`, `SocialSecurity`, `Mortality`). Polymorphic
schemas are sanity-checked structurally but not field-by-field.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from tax_optimizer.config import Config
from tax_optimizer.inputs import (
    CurrentContrib,
    CurrentIncome,
    HealthPremiums,
    Inputs,
    PensionInputs,
    SocialSecurity,
    StartingBalances,
)
from tax_optimizer.mortality import Mortality
from tax_optimizer.scenario import apply_scenario, load_scenario_file


TEMPLATE_PATH = Path("scenarios/template.json")


# ---------------------------------------------------------------------------
# Module-level fixture: parsed JSON for everyone to share.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def template_dict() -> dict:
    return json.loads(TEMPLATE_PATH.read_text())


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


class TestTemplateStructure:
    def test_template_file_exists(self):
        assert TEMPLATE_PATH.exists(), (
            f"{TEMPLATE_PATH} is missing. The template is the canonical "
            "reference for every available scenario knob; it must exist."
        )

    def test_template_is_valid_json(self, template_dict):
        # `load` would raise if invalid; fixture would fail. Reaching
        # here is the assertion.
        assert isinstance(template_dict, dict)

    def test_template_has_config_and_inputs(self, template_dict):
        assert "config" in template_dict
        assert "inputs" in template_dict

    def test_template_parses_through_apply_scenario(self, template_dict):
        """`apply_scenario` validates every key against the dataclass
        fields. If the template has an unknown key, this raises with
        a targeted error message — which is exactly the contract we
        want the template to satisfy."""
        # Start from default Config / Inputs and apply the template.
        # No exception ⇒ every key is recognized.
        apply_scenario(Config(), Inputs(), template_dict)


# ---------------------------------------------------------------------------
# Field-coverage drift checks
# ---------------------------------------------------------------------------


# Polymorphic / structurally-validated blocks. We assert their
# presence but defer field-level coverage to the scenario coercion
# layer (which already validates per-kind).
_POLYMORPHIC_CONFIG_FIELDS = frozenset({"market", "spending"})


def _flat_field_names(dc) -> set[str]:
    return {f.name for f in fields(dc)}


class TestConfigCoverage:
    """Every public field on `Config` must appear in `config` of the
    template (or be one of the documented polymorphic blocks)."""

    def test_every_config_field_is_in_template(self, template_dict):
        cfg_template_keys = set(template_dict["config"])
        cfg_dataclass_fields = _flat_field_names(Config)

        missing = cfg_dataclass_fields - cfg_template_keys
        assert not missing, (
            f"scenarios/template.json is missing config field(s): "
            f"{sorted(missing)}. When you add a new Config knob, "
            f"also add it (with its default value) to the template. "
            f"See tests/test_scenario_template.py docstring for the "
            f"maintenance contract."
        )

    def test_no_extra_config_fields_in_template(self, template_dict):
        cfg_template_keys = set(template_dict["config"])
        cfg_dataclass_fields = _flat_field_names(Config)

        extras = cfg_template_keys - cfg_dataclass_fields
        assert not extras, (
            f"scenarios/template.json has config key(s) that aren't "
            f"on Config: {sorted(extras)}. Either remove them from the "
            f"template or add them to Config."
        )


class TestInputsCoverage:
    """Every public field on `Inputs` (and its nested dataclasses)
    must appear in `inputs` of the template.

    Exception: `Inputs.annual_expenses` is a DEPRECATED back-compat
    field whose simulator path is dead — we deliberately leave it out
    of the template so new scenario files don't propagate the dead
    knob. Adding it back would silently re-introduce it to user
    copy-pasted scenarios.
    """

    _DEPRECATED_INPUTS_FIELDS = frozenset({"annual_expenses"})

    def test_every_inputs_field_is_in_template(self, template_dict):
        inputs_template_keys = set(template_dict["inputs"])
        inputs_dataclass_fields = _flat_field_names(Inputs)

        expected = inputs_dataclass_fields - self._DEPRECATED_INPUTS_FIELDS
        missing = expected - inputs_template_keys
        assert not missing, (
            f"scenarios/template.json is missing inputs field(s): "
            f"{sorted(missing)}. When you add a new Inputs knob, "
            f"also add it (with its default value) to the template."
        )

    def test_no_extra_inputs_fields_in_template(self, template_dict):
        inputs_template_keys = set(template_dict["inputs"])
        inputs_dataclass_fields = _flat_field_names(Inputs)

        extras = inputs_template_keys - inputs_dataclass_fields
        assert not extras, (
            f"scenarios/template.json has inputs key(s) that aren't "
            f"on Inputs: {sorted(extras)}. Either remove them from "
            f"the template or add them to Inputs."
        )

    @pytest.mark.parametrize(
        "block_name, block_cls",
        [
            ("starting", StartingBalances),
            ("income", CurrentIncome),
            ("contrib", CurrentContrib),
            ("pension", PensionInputs),
            ("ss", SocialSecurity),
            ("health_premiums", HealthPremiums),
        ],
    )
    def test_nested_inputs_block_is_complete(
        self, template_dict, block_name, block_cls
    ):
        """Each nested Inputs dataclass must have ALL its public fields
        in the template — that's what makes the template a useful
        starting point for users (they can see every knob without
        running --print-defaults)."""
        block_in_template = template_dict["inputs"].get(block_name, {})
        block_template_keys = set(block_in_template)
        block_dataclass_fields = _flat_field_names(block_cls)

        missing = block_dataclass_fields - block_template_keys
        assert not missing, (
            f"scenarios/template.json inputs.{block_name} is missing "
            f"field(s): {sorted(missing)}. When you add a new field "
            f"to {block_cls.__name__}, also add it to the "
            f"`inputs.{block_name}` block in the template."
        )

        extras = block_template_keys - block_dataclass_fields
        assert not extras, (
            f"scenarios/template.json inputs.{block_name} has key(s) "
            f"that aren't on {block_cls.__name__}: {sorted(extras)}."
        )


class TestMortalityCoverage:
    """`Config.mortality` is its own dataclass. The template's
    mortality sub-dict should expose every field."""

    def test_mortality_block_complete(self, template_dict):
        mortality_in_template = template_dict["config"].get("mortality", {})
        template_keys = set(mortality_in_template)
        dc_fields = _flat_field_names(Mortality)

        missing = dc_fields - template_keys
        assert not missing, (
            f"scenarios/template.json config.mortality is missing "
            f"field(s): {sorted(missing)}."
        )
        extras = template_keys - dc_fields
        assert not extras, (
            f"scenarios/template.json config.mortality has key(s) "
            f"that aren't on Mortality: {sorted(extras)}."
        )


# ---------------------------------------------------------------------------
# Default-value drift: every value in the template should match the
# dataclass default at construction time. Catches typos like
# `"cap_conversion_by_liquidity": false` slipping in (where the field
# default is `True`).
# ---------------------------------------------------------------------------


class TestTemplateValuesMatchDefaults:
    """Every flat (non-polymorphic, non-tagged) value in the template
    should equal the dataclass default. This makes the template
    self-documenting: reading the file IS the spec for "what does
    the simulator do out of the box".
    """

    # Fields whose JSON encoding differs from the dataclass default
    # in expected ways (tax regime stored as string label, etc.).
    _CONFIG_FIELDS_WITH_STRING_ENCODING = frozenset(
        {
            "tax_regime",
            "regime_change_target",
            "state_regime",
            "state_regime_change_target",
        }
    )

    def test_flat_config_defaults_match(self, template_dict):
        cfg = Config()
        for f in fields(Config):
            if f.name in _POLYMORPHIC_CONFIG_FIELDS:
                continue  # market / spending — polymorphic
            if f.name in self._CONFIG_FIELDS_WITH_STRING_ENCODING:
                continue
            if f.name == "mortality":
                continue
            if f.name == "asset_location":
                continue
            template_val = template_dict["config"].get(f.name)
            default_val = getattr(cfg, f.name)
            assert template_val == default_val, (
                f"Template config.{f.name}={template_val!r} disagrees "
                f"with Config default {default_val!r}. Update the "
                f"template OR fix the default."
            )

    def test_flat_inputs_defaults_match(self, template_dict):
        inp = Inputs()
        for f in fields(Inputs):
            if f.name in {
                "starting", "income", "contrib", "pension", "ss",
                "health_premiums",
                "annual_expenses",  # deliberately omitted (deprecated)
            }:
                continue
            template_val = template_dict["inputs"].get(f.name)
            default_val = getattr(inp, f.name)
            assert template_val == default_val, (
                f"Template inputs.{f.name}={template_val!r} disagrees "
                f"with Inputs default {default_val!r}. Update the "
                f"template OR fix the default."
            )

    @pytest.mark.parametrize(
        "block_name, block_cls",
        [
            ("starting", StartingBalances),
            ("income", CurrentIncome),
            ("contrib", CurrentContrib),
            ("pension", PensionInputs),
            ("ss", SocialSecurity),
            ("health_premiums", HealthPremiums),
        ],
    )
    def test_nested_block_defaults_match(
        self, template_dict, block_name, block_cls
    ):
        instance = block_cls()
        block_in_template = template_dict["inputs"].get(block_name, {})
        for f in fields(block_cls):
            template_val = block_in_template.get(f.name)
            default_val = getattr(instance, f.name)
            assert template_val == default_val, (
                f"Template inputs.{block_name}.{f.name}={template_val!r} "
                f"disagrees with {block_cls.__name__} default "
                f"{default_val!r}. Update the template OR fix the default."
            )

    def test_mortality_defaults_match(self, template_dict):
        m = Mortality()
        block = template_dict["config"].get("mortality", {})
        for f in fields(Mortality):
            template_val = block.get(f.name)
            default_val = getattr(m, f.name)
            assert template_val == default_val, (
                f"Template config.mortality.{f.name}={template_val!r} "
                f"disagrees with Mortality default {default_val!r}."
            )


# ---------------------------------------------------------------------------
# End-to-end: the template produces a working simulation.
# ---------------------------------------------------------------------------


class TestTemplateRoundTrip:
    """The template should parse + simulate without error. This is the
    final smoke test that catches any case where the template
    technically validates against the dataclass schema but produces
    a configuration the simulator can't actually execute."""

    def test_template_simulates_without_error(self, template_dict):
        from tax_optimizer.simulator import simulate

        cfg, inputs = apply_scenario(Config(), Inputs(), template_dict)
        df = simulate(cfg, inputs)
        assert len(df) > 0
        # Expected horizon: spouse_a_age_start=50, horizon_age=90 → 41 rows.
        assert len(df) == 41

    def test_template_loads_via_path_api_too(self):
        """`load_scenario_file` is the path-based loader the CLI uses;
        check that path works too."""
        raw = load_scenario_file(TEMPLATE_PATH)
        cfg, inputs = apply_scenario(Config(), Inputs(), raw)
        # Sanity probe on a v6.5 knob.
        assert cfg.cap_conversion_by_liquidity is True
        assert cfg.protect_roth_in_conversion_years is True
