"""Round-trip regression tests for the Dash app's "Download JSON" feature.

Pre-fix the save callback called
``scenario_to_dict(*apply_form_values(values))``, which:

1. Funneled every spending profile through ``_spending_to_dict`` —
   that helper unconditionally emits ``kind="custom"`` with an
   explicit ``phases`` array and a separate ``ltc_shock`` block, even
   for profiles built via :meth:`SpendingProfile.retirement_smile`.
   A user who loaded a "smile" scenario, hit Save, and re-uploaded
   the file would see a "custom" profile they never authored.
2. Walked every dataclass field via ``_inputs_to_dict``, surfacing
   deprecated paths the form schema explicitly hides
   (``inputs.annual_expenses``, ``inputs.ss.start_age``).
3. Re-emitted default-valued config fields the user never set
   (``roth_conversion_amount``, ``section125_reduces_fica_wages``)
   so the diff between an on-disk scenario and its saved counterpart
   was always large and noisy.

These tests pin down the post-fix behavior: the save path now uses
``form_values_to_scenario`` so the JSON shape matches what the form
actually authored.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

pytest.importorskip("dash")

from tax_optimizer import Config, Inputs  # noqa: E402
from tax_optimizer.market import LognormalModel  # noqa: E402
from tax_optimizer.scenario import apply_scenario, scenario_to_dict  # noqa: E402
from tax_optimizer.simulator import simulate  # noqa: E402

from dash_app.state import (  # noqa: E402
    apply_form_values,
    cfg_inputs_to_form_values,
    form_values_to_scenario,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _suppress_deprecation():
    return warnings.catch_warnings()


@pytest.fixture
def example02_dict() -> dict:
    """The user's local scenario file (smile profile, mortality block)."""
    path = REPO_ROOT / "scenarios" / "example02.local.json"
    if not path.exists():
        pytest.skip("scenarios/example02.local.json not present in this checkout")
    return json.loads(path.read_text())


def _save_via_form(cfg: Config, inputs: Inputs) -> dict:
    """Mirror what ``_save_scenario`` does post-fix."""
    form = cfg_inputs_to_form_values(cfg, inputs)
    return form_values_to_scenario(form)


# ---------------------------------------------------------------------
# Bug A: smile profile must NOT downgrade to custom on save
# ---------------------------------------------------------------------


class TestSmileProfilePreserved:
    def test_smile_kind_survives_round_trip(self, example02_dict) -> None:
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), example02_dict)
        saved = _save_via_form(cfg, inputs)
        spending = saved["config"]["spending"]
        assert spending["kind"] == "smile"
        # Smile-only fields the form authors:
        assert spending["base_spending"] == 100_000.0
        assert spending["inflation"] == 0.03
        assert spending["ltc_years"] == 4
        assert spending["ltc_annual_today"] == 150_000.0

    def test_saved_smile_block_omits_custom_only_keys(self, example02_dict) -> None:
        """The form does not author ``phases`` / ``lump_events`` /
        ``ltc_shock``; those must not leak into the saved file."""
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), example02_dict)
        saved = _save_via_form(cfg, inputs)
        spending = saved["config"]["spending"]
        for key in ("phases", "lump_events", "ltc_shock"):
            assert key not in spending, (
                f"{key!r} leaked into a smile-shaped save; "
                f"the scenario decoder will reject it via "
                f"_check_keys('spending(smile)', ...)."
            )

    def test_saved_file_re_loads_cleanly(self, example02_dict) -> None:
        """The saved JSON must pass the strict ``_check_keys`` decoder."""
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), example02_dict)
        saved = _save_via_form(cfg, inputs)
        # Round-trip the JSON the way `dcc.Upload` would:
        text = json.dumps(saved, indent=2, default=str)
        reloaded = json.loads(text)
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg_back, inputs_back = apply_scenario(
                Config(), Inputs(), reloaded
            )
        # Sim equivalence: identical first-year + last-year balance
        # vector.
        df_orig = simulate(cfg, inputs)
        df_back = simulate(cfg_back, inputs_back)
        balance_cols = [
            "pretax_balance",
            "roth_balance",
            "taxable_balance",
            "hsa_balance",
        ]
        for col in balance_cols:
            assert df_orig.iloc[0][col] == pytest.approx(df_back.iloc[0][col])
            assert df_orig.iloc[-1][col] == pytest.approx(df_back.iloc[-1][col])


class TestFlatProfilePreserved:
    def test_flat_kind_survives_round_trip(self) -> None:
        scn = {
            "config": {
                "spending": {
                    "kind": "flat",
                    "base_spending": 65_000.0,
                    "inflation": 0.025,
                },
            },
        }
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), scn)
        saved = _save_via_form(cfg, inputs)
        spending = saved["config"]["spending"]
        assert spending["kind"] == "flat"
        assert spending["base_spending"] == 65_000.0
        for key in ("phases", "lump_events", "ltc_shock", "ltc_years",
                    "ltc_annual_today"):
            assert key not in spending


# ---------------------------------------------------------------------
# Bug A: deprecated / legacy paths must not leak into saved JSON
# ---------------------------------------------------------------------


class TestHiddenPathsNotEmitted:
    def test_inputs_annual_expenses_not_emitted(self, example02_dict) -> None:
        """`inputs.annual_expenses` is deprecated in favor of
        `config.annual_expenses_today` / `config.spending.base_spending`.
        The form does not surface it; the save must not either."""
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), example02_dict)
        saved = _save_via_form(cfg, inputs)
        assert "annual_expenses" not in saved.get("inputs", {})

    def test_inputs_ss_start_age_not_emitted(self, example02_dict) -> None:
        """`inputs.ss.start_age` is the legacy single-spouse SS claim
        age; we expose ``start_age_a`` / ``start_age_b`` instead."""
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), example02_dict)
        saved = _save_via_form(cfg, inputs)
        ss = saved.get("inputs", {}).get("ss", {})
        assert "start_age" not in ss
        # The per-spouse fields *must* still be present.
        assert ss["start_age_a"] == 70
        assert ss["start_age_b"] == 65


# ---------------------------------------------------------------------
# Bug A: market discriminator stays accurate (kind+fields aligned)
# ---------------------------------------------------------------------


class TestMarketRoundTrip:
    def test_lognormal_block_only_carries_lognormal_fields(self) -> None:
        scn = {
            "config": {
                "market": {
                    "kind": "lognormal",
                    "equity_mu": 0.06,
                    "equity_sigma": 0.16,
                    "bond_mu": 0.035,
                    "bond_sigma": 0.05,
                    "equity_bond_corr": 0.05,
                },
            },
        }
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), scn)
        saved = _save_via_form(cfg, inputs)
        market = saved["config"]["market"]
        assert market["kind"] == "lognormal"
        assert market["equity_mu"] == 0.06
        # No deterministic-only keys leaked in:
        assert "equity" not in market
        assert "bond" not in market
        assert "block_size" not in market

    def test_deterministic_block_carries_deterministic_fields(self) -> None:
        scn = {
            "config": {
                "market": {
                    "kind": "deterministic",
                    "equity": 0.05,
                    "bond": 0.025,
                },
            },
        }
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), scn)
        saved = _save_via_form(cfg, inputs)
        market = saved["config"]["market"]
        assert market["kind"] == "deterministic"
        assert market["equity"] == 0.05
        assert market["bond"] == 0.025
        # No lognormal-only keys leaked in:
        for key in ("equity_mu", "equity_sigma", "bond_mu", "bond_sigma",
                    "equity_bond_corr", "cape_today", "cape_long_run"):
            assert key not in market


# ---------------------------------------------------------------------
# Bug B: scenario_to_dict must preserve cape_long_run when cape_today
# is None.
# ---------------------------------------------------------------------


class TestCapeLongRunPreserved:
    def test_cape_long_run_emitted_when_cape_today_unset(self) -> None:
        """Pre-fix `_market_to_dict` only emitted `cape_long_run`
        inside the ``cape_today is not None`` branch, silently
        dropping a user-set value during JSON round-trip."""
        m = LognormalModel(
            equity_mu=0.07,
            equity_sigma=0.18,
            bond_mu=0.04,
            bond_sigma=0.06,
            equity_bond_corr=0.10,
            cape_today=None,
            cape_long_run=22.0,
        )
        cfg = Config(market=m)
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            d = scenario_to_dict(cfg, Inputs())
        market = d["config"]["market"]
        # cape_today stays absent (its absence is the documented
        # "no scaling" sentinel; a literal `null` would round-trip
        # the same).
        assert market.get("cape_today") in (None, ...) or "cape_today" not in market
        # cape_long_run must round-trip.
        assert market["cape_long_run"] == 22.0

    def test_cape_long_run_round_trips_through_apply_scenario(self) -> None:
        m = LognormalModel(
            equity_mu=0.07,
            equity_sigma=0.18,
            bond_mu=0.04,
            bond_sigma=0.06,
            equity_bond_corr=0.10,
            cape_today=None,
            cape_long_run=22.0,
        )
        cfg = Config(market=m)
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            d = scenario_to_dict(cfg, Inputs())
            cfg_back, _ = apply_scenario(Config(), Inputs(), d)
        assert isinstance(cfg_back.market, LognormalModel)
        assert cfg_back.market.cape_today is None
        assert cfg_back.market.cape_long_run == 22.0

    def test_cape_today_set_still_emits_both(self) -> None:
        """Sanity that the original behavior still works when
        ``cape_today`` is set (the fix only added the unconditional
        ``cape_long_run`` emit)."""
        m = LognormalModel(
            equity_mu=0.07,
            equity_sigma=0.18,
            bond_mu=0.04,
            bond_sigma=0.06,
            equity_bond_corr=0.10,
            cape_today=33.0,
            cape_long_run=16.5,
        )
        cfg = Config(market=m)
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            d = scenario_to_dict(cfg, Inputs())
        market = d["config"]["market"]
        assert market["cape_today"] == 33.0
        assert market["cape_long_run"] == 16.5


# ---------------------------------------------------------------------
# End-to-end: save -> reload -> save must reach a fixed point
# ---------------------------------------------------------------------


class TestSaveReloadFixedPoint:
    def test_two_saves_byte_identical(self, example02_dict) -> None:
        """Mirrors the canonical user workflow: load file, save it
        back out, and the second save (after re-upload) should be
        byte-identical to the first. Pre-fix this was true only
        because both saves emitted the same noisy custom-shaped
        JSON; post-fix it should also be true and *also* match the
        form's authored shape."""
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg1, inputs1 = apply_scenario(Config(), Inputs(), example02_dict)
        saved1 = _save_via_form(cfg1, inputs1)
        text1 = json.dumps(saved1, indent=2, default=str)

        # Reload the saved JSON and save again.
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg2, inputs2 = apply_scenario(Config(), Inputs(), json.loads(text1))
        saved2 = _save_via_form(cfg2, inputs2)
        text2 = json.dumps(saved2, indent=2, default=str)

        assert text1 == text2, "save -> reload -> save must reach a fixed point"

    def test_apply_form_values_validates_save_payload(
        self, example02_dict
    ) -> None:
        """The save callback uses ``apply_form_values(values)`` as
        a validation step before emitting the dict. This test pins
        down that the form values for example02 round-trip through
        the decoder without error."""
        with _suppress_deprecation():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg, inputs = apply_scenario(Config(), Inputs(), example02_dict)
            form = cfg_inputs_to_form_values(cfg, inputs)
            # Must not raise:
            cfg_back, inputs_back = apply_form_values(form)
        # And the resulting cfg/inputs must produce identical
        # simulator output.
        df1 = simulate(cfg, inputs)
        df2 = simulate(cfg_back, inputs_back)
        assert df1.iloc[-1]["pretax_balance"] == pytest.approx(
            df2.iloc[-1]["pretax_balance"]
        )
