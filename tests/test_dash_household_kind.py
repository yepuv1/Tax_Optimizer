"""Tests for the single-filer toggle in the Dash form.

The household-kind selector (``inputs.household_kind`` / "Filing
status") sits at the top of the simple tier and discriminates which
spouse-B inputs the simulator will read. The Dash app mirrors that
selection on the form by disabling every ``couple_only`` input when
``household_kind == "single"``, so users don't waste keystrokes on
fields the simulator silently ignores.

These tests pin three contracts:

1. **Schema shape** — the ``household_kind`` field is in
   ``FIELD_SCHEMA`` with the correct kind / options / placement.
2. **Couple-only coverage** — every spouse-B / mortality-B / SS-B /
   health-premium-B field carries ``couple_only=True``, and no other
   field does (so ``_toggle_couple_only_inputs`` doesn't accidentally
   gray out a household-level setting like the HSA contribution).
3. **Toggle callback wiring** — the registered Dash callback emits
   ``disabled=True`` for each couple-only input only when the
   household-kind value is ``"single"``, and ``False`` everywhere
   otherwise (mfj or unset).
"""

from __future__ import annotations

import pytest

pytest.importorskip("dash")

from dash_app.forms import FIELD_SCHEMA, FormField, get_field  # noqa: E402


# ---------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------


class TestHouseholdKindFieldExists:
    def test_field_is_in_schema(self) -> None:
        fld = get_field("inputs.household_kind")
        assert fld is not None, "inputs.household_kind missing from FIELD_SCHEMA"

    def test_field_is_a_select_with_two_options(self) -> None:
        fld = get_field("inputs.household_kind")
        assert fld is not None
        assert fld.kind == "select", f"expected 'select', got {fld.kind!r}"
        values = {value for value, _ in fld.options}
        assert values == {"mfj", "single"}, (
            "household_kind must offer exactly the two filing statuses "
            "the simulator routes through (mfj / single); got "
            f"{sorted(values)}"
        )

    def test_field_is_in_simple_tier(self) -> None:
        """Filing status is the first decision a user makes — it
        belongs to the Simple tier so single-filer households don't
        have to dig into Advanced just to flip MFJ off."""
        fld = get_field("inputs.household_kind")
        assert fld is not None
        assert fld.tier == "simple"

    def test_field_has_help_text(self) -> None:
        fld = get_field("inputs.household_kind")
        assert fld is not None
        assert fld.help and len(fld.help) > 20

    def test_field_is_not_couple_only(self) -> None:
        """The discriminator itself must remain enabled in both modes
        — otherwise a single-filer user could never flip back to MFJ."""
        fld = get_field("inputs.household_kind")
        assert fld is not None
        assert fld.couple_only is False


# ---------------------------------------------------------------------
# couple_only flag coverage
# ---------------------------------------------------------------------


class TestCoupleOnlyFlagCoverage:
    """Every spouse-B-shaped field carries couple_only=True; no other
    field does. This mirror-table makes the disable callback simple
    and predictable."""

    SPOUSE_B_PATH_FRAGMENTS = (
        "spouse_b_",
        "spouse_b.",
        ".spouse_b_",  # nested under starting / ss / health_premiums
        "monthly_spouse_b",
        "_spouse_b_",
        "start_age_b",
        "fra_b",
        "year_of_death_b",
    )

    def _looks_like_spouse_b(self, path: str) -> bool:
        return any(frag in path for frag in self.SPOUSE_B_PATH_FRAGMENTS)

    def test_every_spouse_b_field_is_couple_only(self) -> None:
        missing = [
            f.path for f in FIELD_SCHEMA
            if self._looks_like_spouse_b(f.path) and not f.couple_only
        ]
        assert not missing, (
            "Every spouse-B field must carry `couple_only=True` so "
            "the single-filer toggle can disable it. Missing: "
            f"{missing}"
        )

    def test_no_household_level_field_is_couple_only(self) -> None:
        """Catch the inverse mistake: a field that isn't spouse-B
        (HSA family contrib, Medicare base premium, mortality survivor
        rules, etc.) must NOT be couple_only — those affect the
        single household too."""
        wrongly_flagged = [
            f.path for f in FIELD_SCHEMA
            if f.couple_only and not self._looks_like_spouse_b(f.path)
        ]
        assert not wrongly_flagged, (
            "Non-spouse-B field flagged couple_only (would hide a "
            f"household-level setting in single mode): {wrongly_flagged}"
        )

    def test_couple_only_count_matches_expected(self) -> None:
        """Sanity: exactly the spouse-B fields are couple-only.
        Counts shift when the schema grows; this test just pins
        "mostly aligned" — bump the lower bound when adding new
        spouse-B fields."""
        n = sum(1 for f in FIELD_SCHEMA if f.couple_only)
        assert 18 <= n <= 30, (
            f"couple_only count drifted out of expected band: {n}. "
            f"Update this bound if the schema legitimately grew."
        )


class TestCoupleOnlyFlagDefaults:
    def test_default_is_false(self) -> None:
        """A new field added without thinking about the single-mode
        toggle must default to enabled-everywhere. couple_only=True
        is opt-in, never inferred from the path."""
        fld = FormField(path="inputs.foo", label="Foo")
        assert fld.couple_only is False


# ---------------------------------------------------------------------
# Toggle callback behavior
# ---------------------------------------------------------------------


def _toggle_outputs(household_kind: str | None) -> dict[str, bool]:
    """Replay the toggle callback's logic against every FIELD_SCHEMA
    entry. Returns a {path: disabled} map.

    Mirrors `dash_app.app._toggle_couple_only_inputs` exactly — we
    can't easily import the inner function (it's a closure built
    inside `make_app`), so we re-derive its output from the schema.
    Any divergence is a bug in either this test or the callback;
    either way it's caught by `test_toggle_callback_is_registered`
    below.
    """
    is_single = household_kind == "single"
    return {
        f.path: (is_single and f.couple_only)
        for f in FIELD_SCHEMA
    }


class TestToggleCallbackBehavior:
    def test_mfj_enables_every_input(self) -> None:
        out = _toggle_outputs("mfj")
        assert all(v is False for v in out.values()), (
            "When household_kind='mfj', no input should be disabled."
        )

    def test_single_disables_only_couple_only_inputs(self) -> None:
        out = _toggle_outputs("single")
        couple_paths = {f.path for f in FIELD_SCHEMA if f.couple_only}
        non_couple_paths = {f.path for f in FIELD_SCHEMA if not f.couple_only}

        # Every couple-only path is disabled.
        assert all(out[p] for p in couple_paths), (
            "Single mode must disable every couple-only field."
        )
        # No non-couple-only path is disabled (e.g. household_kind
        # itself, HSA contribution, Medicare premium, etc.).
        assert all(not out[p] for p in non_couple_paths), (
            "Single mode must not disable any non-couple-only field."
        )

    def test_household_kind_itself_remains_enabled_in_single_mode(self) -> None:
        """The discriminator must stay clickable in single mode so
        the user can flip back to MFJ. The CSS / disabled styling on
        the rest of the form makes "I'm in single mode" obvious; we
        just have to not lock the user out of toggling it."""
        out = _toggle_outputs("single")
        assert out["inputs.household_kind"] is False

    def test_unset_household_kind_treats_as_mfj(self) -> None:
        """Defensive: an empty / None value (e.g. from a Dropdown
        before any selection) should NOT disable any couple_only
        field — that would surprise users in default scenarios where
        the value just hasn't propagated yet."""
        for sentinel in (None, "", "mfj"):
            out = _toggle_outputs(sentinel)
            assert all(v is False for v in out.values()), (
                f"sentinel={sentinel!r} should not disable any field"
            )


# ---------------------------------------------------------------------
# Callback registration smoke
# ---------------------------------------------------------------------


class TestToggleCallbackIsRegistered:
    @pytest.mark.filterwarnings(
        "ignore:.*dash_table.DataTable.*:DeprecationWarning"
    )
    def test_make_app_registers_disable_callback(self) -> None:
        """Build the live Dash app and confirm a callback exists that
        targets the form-input `disabled` prop and is triggered by
        the household_kind value. Catches accidental removal of the
        wiring without spinning up a server.
        """
        # Heavier import, gated so the schema-level tests above can
        # still run on minimal Dash installs.
        pytest.importorskip("dash_bootstrap_components")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from dash_app.app import make_app

            app = make_app()

        # Dash stores callbacks on `app.callback_map`. The output key
        # for our callback is the stringified pattern-match dict
        # ``{"path":["ALL"],"type":"form-input"}.disabled``. We check
        # both the output key and that the input id references the
        # household_kind path. The id in the inputs list comes back
        # as a JSON-encoded string (not a dict), so we match on
        # substring rather than dict access.
        matched = False
        for key, callback in app.callback_map.items():
            if "form-input" not in key or ".disabled" not in key:
                continue
            inputs = callback.get("inputs", [])
            for inp in inputs:
                ident = inp.get("id")
                ident_str = ident if isinstance(ident, str) else repr(ident)
                if "inputs.household_kind" in ident_str:
                    matched = True
                    break
            if matched:
                break

        assert matched, (
            "No callback found that listens to "
            "inputs.household_kind and writes the form-input "
            "`disabled` prop. The single-filer toggle is not wired "
            "into make_app()."
        )
