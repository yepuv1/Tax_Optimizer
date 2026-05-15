"""Form-input rendering regressions for the Dash app.

Pre-fix the form inputs rendered with a numeric `step` attribute that
the browser used as a *validation* rule (`(value - min) % step == 0`),
which caused two visible regressions:

1. **Percent fields** with `step=0.005` were marked `:invalid` for
   legitimate values like `0.07` because of float-point arithmetic
   (`0.07 / 0.005 == 14.000000000000002`, not 14). Chrome / Safari /
   Firefox all paint `:invalid` `<input>` text red.
2. **Number fields** without an explicit step fell through to the
   HTML5 default `step=1`, marking any non-integer dollar amount
   (e.g. `taxable_brokerage=460270.90`) as `:invalid` -> red.
   Even an explicit `step=1000` did not save us because real-world
   balances aren't exact multiples of 1000.

The fix is `step="any"` for both `percent` and `number` kinds, which
disables HTML5 step validation while keeping `min`/`max` bounds
intact. `int` keeps its `step=1` (integer-only is intentional).
"""

from __future__ import annotations

import pytest

pytest.importorskip("dash")

from dash_app.forms import FIELD_SCHEMA, get_field  # noqa: E402
from dash_app.layout import _input_for  # noqa: E402


def _form_inputs():
    """Yield every rendered form-input (path, type, step, min, max).

    We invoke ``_input_for`` directly per schema field rather than
    standing up the whole ``make_app()`` layout — that path imports
    ``dash.dash_table.DataTable`` which emits a ``DeprecationWarning``
    that ``filterwarnings = ["error"]`` (in ``pyproject.toml``) would
    promote to a test failure. The control we're testing here is
    purely the input renderer, so this is the more focused harness.
    """
    for fld in FIELD_SCHEMA:
        # Use a representative value; the actual value doesn't affect
        # the step / min / max attributes the renderer emits.
        if fld.kind == "bool":
            continue  # bools render as a Switch, not an Input
        if fld.kind == "select":
            continue  # selects render as a Dropdown, not an Input
        c = _input_for(fld, value=None)
        if c.__class__.__name__ != "Input":
            continue
        yield (
            fld.path,
            getattr(c, "type", None),
            getattr(c, "step", None),
            getattr(c, "min", None),
            getattr(c, "max", None),
        )


# ---------------------------------------------------------------------
# Step attribute hygiene
# ---------------------------------------------------------------------


class TestPercentInputsAllowAnyStep:
    """Every `percent` field must render with `step="any"` so values
    like `0.07` (not divisible by 0.005 in float arithmetic) don't
    trigger `:invalid` red text."""

    def test_all_percent_inputs_use_step_any(self) -> None:
        bad = []
        for path, _, step, _, _ in _form_inputs():
            fld = get_field(path)
            if fld is None or fld.kind != "percent":
                continue
            if step != "any":
                bad.append((path, step))
        assert not bad, (
            f"{len(bad)} percent field(s) still carry a numeric step that "
            f"will trigger HTML5 :invalid styling on float-point inputs:\n"
            + "\n".join(f"  {p}: step={s!r}" for p, s in bad)
        )

    def test_percent_step_is_string_not_float(self) -> None:
        """Sanity: `step=0.005` (the pre-fix value) is float, `"any"`
        is str. Catch a regression that uses 0.005 again, even
        coincidentally."""
        for path, _, step, _, _ in _form_inputs():
            fld = get_field(path)
            if fld is None or fld.kind != "percent":
                continue
            assert isinstance(step, str), (
                f"percent input {path!r} has non-string step={step!r}; "
                f"expected the literal string 'any'."
            )


class TestNumberInputsAllowAnyStep:
    """`number` kinds (currency, dollar amounts) must also use
    `step="any"` because real-world values like `460270.90` aren't
    multiples of any sensible spinner increment."""

    def test_all_number_inputs_use_step_any(self) -> None:
        bad = []
        for path, _, step, _, _ in _form_inputs():
            fld = get_field(path)
            if fld is None or fld.kind != "number":
                continue
            if step != "any":
                bad.append((path, step))
        assert not bad, (
            f"{len(bad)} number field(s) still carry a numeric step. "
            f"Real-world dollar amounts like $460,270.90 are not "
            f"exact multiples of any spinner increment, so any non-'any' "
            f"step will trigger :invalid red text:\n"
            + "\n".join(f"  {p}: step={s!r}" for p, s in bad)
        )


class TestIntInputsKeepIntegerStep:
    """`int` fields are integer-only by intent (ages, year offsets,
    horizon, etc.). They should keep `step=1` (the default) so the
    spinner increments by 1 and the browser correctly rejects e.g.
    "65.5"."""

    def test_int_inputs_use_step_1(self) -> None:
        for path, _, step, _, _ in _form_inputs():
            fld = get_field(path)
            if fld is None or fld.kind != "int":
                continue
            assert step == (fld.step or 1), (
                f"int input {path!r} has step={step!r}; expected 1 "
                f"(or the schema's explicit step={fld.step!r})."
            )

    def test_at_least_one_int_input_exists(self) -> None:
        """Sanity: the assertion above is non-vacuous."""
        ints = [
            p for p, _, _, _, _ in _form_inputs()
            if (f := get_field(p)) is not None and f.kind == "int"
        ]
        assert len(ints) >= 5


# ---------------------------------------------------------------------
# Min / max are still honored — the fix only disabled `step` validation.
# ---------------------------------------------------------------------


class TestBoundsArePreserved:
    def test_age_inputs_still_carry_min_max_bounds(self) -> None:
        """The fix should not have stripped `min`/`max` — only `step`.
        Spouse-age fields are a stable example with explicit bounds."""
        bounds = {
            path: (mn, mx)
            for path, _, _, mn, mx in _form_inputs()
            if path == "inputs.spouse_a_age_start"
        }
        assert bounds, "expected to find inputs.spouse_a_age_start"
        mn, mx = next(iter(bounds.values()))
        assert mn == 18 and mx == 100


# ---------------------------------------------------------------------
# Concrete reproduction of the original red-text values
# ---------------------------------------------------------------------


class TestKnownInvalidValuesNowAccepted:
    """The values that triggered the original red-text complaint must
    no longer satisfy the (no-longer-applied) step constraint, but
    crucially the inputs no longer *carry* a step that would trip
    them. We assert via a Python-side simulation of the HTML5 check."""

    @pytest.mark.parametrize(
        "value,step,reason",
        [
            (0.07, 0.005, "0.07 / 0.005 = 14.000000000000002 (float drift)"),
            (460_270.90, 1000, "real-world taxable balance, non-1000-multiple"),
            (1_549_863.75, 1000, "real-world 401(k) balance, non-1000-multiple"),
            (416_741.12, 1000, "real-world pension balance, non-1000-multiple"),
        ],
    )
    def test_known_invalid_values_would_have_been_flagged(
        self, value, step, reason
    ) -> None:
        """Sanity check that the inputs the user actually enters DO
        violate the old step rule. Demonstrates *why* the fix is
        needed; the step is now `"any"` so the rule no longer applies.
        """
        q = value / step
        assert q != int(q), (
            f"Value {value} is unexpectedly a clean multiple of {step}; "
            f"reason: {reason}"
        )
