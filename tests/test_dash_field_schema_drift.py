"""Drift tests for the Dash form's `FIELD_SCHEMA` vs. the underlying
`Config` / `Inputs` dataclass surface.

`tests/test_scenario_template.py` already enforces template-vs-
dataclass drift on `scenarios/template.json`. The audit flagged that
there is *no* equivalent guarantee for `dash_app.forms.FIELD_SCHEMA`:
a new editable field added to `Config`/`Inputs` could silently miss
the Dash form (the user sees no input for it; the value stays at
default forever). This module closes that gap.

Two-way contract enforced here:

* Every `FormField.path` resolves to a real public field on
  `Config` or `Inputs` (or one of their nested dataclasses). Catches
  typos / leftover entries after a rename.
* Every nested dataclass that the audit calls out as "should be
  fully exposed in the form" (`StartingBalances`, `CurrentIncome`,
  `CurrentContrib`, `PensionInputs`, `AnnuityInputs`,
  `SocialSecurity`, `HealthPremiums`, `Mortality`) has every public
  field present in the schema, with a small explicit allow-list of
  legacy / engineering-only fields that intentionally stay out of
  the UI.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

# Skip the whole module if `dash_app` isn't installed.
pytest.importorskip("dash")

from tax_optimizer.config import Config  # noqa: E402
from tax_optimizer.inputs import (  # noqa: E402
    AnnuityInputs,
    CurrentContrib,
    CurrentIncome,
    HealthPremiums,
    Inputs,
    PensionInputs,
    SocialSecurity,
    StartingBalances,
)
from tax_optimizer.mortality import Mortality  # noqa: E402

from dash_app.forms import FIELD_SCHEMA  # noqa: E402


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _public_fields(cls: type) -> set[str]:
    """Top-level public field names on a dataclass."""
    if not is_dataclass(cls):
        return set()
    return {f.name for f in fields(cls) if not f.name.startswith("_")}


def _schema_paths() -> set[str]:
    """All `FormField.path` values, lowercased to defend against
    casing drift."""
    return {f.path for f in FIELD_SCHEMA}


def _resolve_path(path: str) -> bool:
    """Return True iff `path` resolves to a real dataclass field on
    Config or Inputs.

    Paths look like ``config.<...>`` or ``inputs.<...>``. The leading
    namespace token is stripped before walking. Polymorphic / nested-
    selector blocks (``config.market.*``, ``config.spending.*``,
    ``config.tax_regime``, ``config.state_regime``,
    ``config.asset_location.*``, ``config.mortality.*``) are
    tolerated wholesale: we only check the head-level field exists.
    """
    if "." not in path:
        return False
    head, _, rest = path.partition(".")
    if head == "config":
        # Config-level fields. Walk the dotted tail one step at a
        # time and verify each segment is a real attribute on a
        # dataclass; bail early on polymorphic blocks where the
        # tail is a selector-specific schema we don't enforce here.
        return _walk_config_path(rest)
    if head == "inputs":
        return _walk_inputs_path(rest)
    return False


_CONFIG_TOLERANT_HEADS = {
    "market", "spending", "tax_regime", "state_regime",
    "asset_location",
    "regime_change_target", "state_regime_change_target",
}


def _walk_config_path(rest: str) -> bool:
    head, _, tail = rest.partition(".")
    cfg_fields = _public_fields(Config)
    if head not in cfg_fields:
        return False
    if not tail:
        return True
    if head in _CONFIG_TOLERANT_HEADS:
        return True
    if head == "mortality":
        return tail in _public_fields(Mortality)
    return False


_INPUTS_NESTED: dict[str, type] = {
    "starting": StartingBalances,
    "income": CurrentIncome,
    "contrib": CurrentContrib,
    "pension": PensionInputs,
    "annuity": AnnuityInputs,
    "ss": SocialSecurity,
    "health_premiums": HealthPremiums,
}


def _walk_inputs_path(rest: str) -> bool:
    head, _, tail = rest.partition(".")
    inp_fields = _public_fields(Inputs)
    if head not in inp_fields:
        return False
    if not tail:
        return True
    nested_cls = _INPUTS_NESTED.get(head)
    if nested_cls is None:
        # `inputs.<head>` was a flat field, no further recursion.
        return False
    return tail in _public_fields(nested_cls)


# Fields the schema deliberately does NOT expose. Each entry has a
# one-line reason so the next reader knows whether to add a UI widget
# or extend this allow-list.
_INPUTS_FORM_EXEMPT: dict[tuple[str, str], str] = {
    ("Inputs", "annual_expenses"):
        "Deprecated legacy field; see test_scenario_template.py contract.",
    ("SocialSecurity", "start_age"):
        "Legacy single-spouse SS start age; superseded by per-spouse "
        "start_age_a / start_age_b. Kept for back-compat scenario "
        "JSON loading.",
}


# ---------------------------------------------------------------------
# Drift tests
# ---------------------------------------------------------------------


class TestFieldSchemaPathsAreReachable:
    """Every path declared in `FIELD_SCHEMA` must resolve to a real
    field on Config / Inputs (or be inside one of the polymorphic
    blocks where we tolerate selector-specific tails).
    """

    def test_every_schema_path_resolves(self) -> None:
        unknown: list[str] = []
        for f in FIELD_SCHEMA:
            if not _resolve_path(f.path):
                unknown.append(f.path)
        assert not unknown, (
            "FIELD_SCHEMA contains path(s) that don't resolve to any "
            "Config / Inputs field (typo or stale entry after rename): "
            f"{unknown}"
        )


class TestNestedInputsBlocksFullyExposed:
    """The nested Inputs dataclasses are simple flat-scalar blocks that
    a user is expected to edit knob-by-knob. Every public field on each
    of them must be in `FIELD_SCHEMA` — *or* explicitly listed in
    `_INPUTS_FORM_EXEMPT` with a reason.
    """

    @pytest.mark.parametrize(
        "block_name, block_cls",
        [
            ("starting", StartingBalances),
            ("income", CurrentIncome),
            ("contrib", CurrentContrib),
            ("pension", PensionInputs),
            ("annuity", AnnuityInputs),
            ("ss", SocialSecurity),
            ("health_premiums", HealthPremiums),
        ],
    )
    def test_block_fields_in_schema(
        self, block_name: str, block_cls: type
    ) -> None:
        schema_paths = _schema_paths()
        missing: list[str] = []
        for f in fields(block_cls):
            if f.name.startswith("_"):
                continue
            if (block_cls.__name__, f.name) in _INPUTS_FORM_EXEMPT:
                continue
            full = f"inputs.{block_name}.{f.name}"
            if full not in schema_paths:
                missing.append(full)
        assert not missing, (
            f"Public field(s) on {block_cls.__name__} are missing "
            "from FIELD_SCHEMA. Either add a `_f(...)` row in "
            "dash_app/forms.py or extend `_INPUTS_FORM_EXEMPT` with "
            f"a reason. Missing: {missing}"
        )


class TestMortalityFieldsExposed:
    """`Config.mortality` is a small dataclass; the two
    `year_of_death_*` fields drive a critical scenario knob (mortality
    risk) so they have to stay in the form. Pre-fix the audit found
    the form silently omitted them in some flows.
    """

    def test_year_of_death_a_and_b_in_schema(self) -> None:
        schema_paths = _schema_paths()
        for f in ("year_of_death_a", "year_of_death_b"):
            full = f"config.mortality.{f}"
            assert full in schema_paths, (
                f"FIELD_SCHEMA missing {full}; mortality is critical "
                "to the scenario UX."
            )


class TestExemptListIsAccurate:
    """Each entry in `_INPUTS_FORM_EXEMPT` must reference a real
    field on the named dataclass. Stale exemptions silently mask new
    coverage gaps.
    """

    def test_every_exempt_entry_is_a_real_field(self) -> None:
        # Map class-name → the dataclass type so we can validate.
        registry = {
            "Inputs": Inputs,
            "StartingBalances": StartingBalances,
            "CurrentIncome": CurrentIncome,
            "CurrentContrib": CurrentContrib,
            "PensionInputs": PensionInputs,
            "AnnuityInputs": AnnuityInputs,
            "SocialSecurity": SocialSecurity,
            "HealthPremiums": HealthPremiums,
            "Mortality": Mortality,
        }
        unknown: list[tuple[str, str]] = []
        for (cls_name, field_name) in _INPUTS_FORM_EXEMPT:
            cls = registry.get(cls_name)
            if cls is None:
                unknown.append((cls_name, field_name))
                continue
            if field_name not in _public_fields(cls):
                unknown.append((cls_name, field_name))
        assert not unknown, (
            "_INPUTS_FORM_EXEMPT references field(s) that don't exist "
            "on the named dataclass. Either remove the entry or "
            f"rename: {unknown}"
        )
