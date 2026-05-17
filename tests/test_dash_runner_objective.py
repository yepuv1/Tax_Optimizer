"""Regression tests for the audit's "expose optimizer objective +
n_paths/maxiter/popsize in Dash run controls" finding.

The Dash app's `runner.run_scenario` now accepts ``objective``,
``maxiter``, and ``popsize`` kwargs that flow into the differential-
evolution optimizer used to build S3. Pre-fix these were hard-coded
at the runner level (``terminal`` / 20 / 10) so the Dash UI could not
target CVaR or probability-of-success objectives even though the
underlying ``optimizer.optimize_household`` supports them.

These tests pin:

  * the new args reach ``optimize_household`` via a spy patch,
  * invalid objective / maxiter / popsize values raise ``ValueError``,
  * the legacy 0-arg invocation still works (defaults preserved),
  * the Dash layout exposes the three new controls so the
    callback-graph wiring stays intact.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from unittest.mock import patch

import pytest


@contextmanager
def _silence_dash_deprecations():
    """`dash_table.DataTable` triggers a DeprecationWarning when the
    layout instantiates it. The pytest config treats warnings as
    errors, so we have to scope-silence around `build_layout`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield

# `dash_app` is an optional install. Skip the whole module if it's
# absent so the base test suite still passes.
pytest.importorskip("dash")

from tax_optimizer import Config, Inputs  # noqa: E402

from dash_app.runner import _build_four, run_scenario  # noqa: E402


def _small() -> tuple[Config, Inputs]:
    return Config(horizon_age=58), Inputs()


class TestRunnerForwardsObjective:
    def test_default_objective_is_terminal(self) -> None:
        cfg, inp = _small()
        with patch(
            "dash_app.runner.optimize_household",
            return_value=(cfg, inp, {}),
        ) as m:
            run_scenario(cfg, inp, mode="four_strategies", seed=0)
        kwargs = m.call_args.kwargs
        assert kwargs["objective"] == "terminal"
        assert kwargs["maxiter"] == 20
        assert kwargs["popsize"] == 10

    def test_cvar_objective_passed_through(self) -> None:
        cfg, inp = _small()
        with patch(
            "dash_app.runner.optimize_household",
            return_value=(cfg, inp, {}),
        ) as m:
            run_scenario(
                cfg, inp,
                mode="four_strategies",
                seed=0,
                objective="cvar",
                maxiter=5,
                popsize=8,
            )
        kwargs = m.call_args.kwargs
        assert kwargs["objective"] == "cvar"
        assert kwargs["maxiter"] == 5
        assert kwargs["popsize"] == 8

    def test_p_success_objective_passed_through(self) -> None:
        cfg, inp = _small()
        with patch(
            "dash_app.runner.optimize_household",
            return_value=(cfg, inp, {}),
        ) as m:
            run_scenario(
                cfg, inp,
                mode="four_strategies",
                seed=0,
                objective="p_success",
            )
        kwargs = m.call_args.kwargs
        assert kwargs["objective"] == "p_success"


class TestValidation:
    def test_unknown_objective_raises(self) -> None:
        cfg, inp = _small()
        with pytest.raises(ValueError, match="objective"):
            _build_four(cfg, inp, seed=0, objective="bogus")

    def test_maxiter_below_one_raises(self) -> None:
        cfg, inp = _small()
        with pytest.raises(ValueError, match="maxiter"):
            _build_four(cfg, inp, seed=0, maxiter=0)

    def test_popsize_below_four_raises(self) -> None:
        # SciPy differential_evolution requires popsize >= 4.
        cfg, inp = _small()
        with pytest.raises(ValueError, match="popsize"):
            _build_four(cfg, inp, seed=0, popsize=3)

    def test_opt_n_paths_below_one_raises(self) -> None:
        cfg, inp = _small()
        with pytest.raises(ValueError, match="opt_n_paths"):
            _build_four(cfg, inp, seed=0, opt_n_paths=0)


class TestOptimizerInternalNPaths:
    """The optimizer's *inner* MC path count (`opt_n_paths`) is
    distinct from the post-optimization fan-chart path count
    (`n_paths`). The latter ran-once-per-strategy can afford 200-2000
    paths; the former runs hundreds of times inside the DE loop and
    must stay small or the Dash UI hangs.
    """

    def test_default_opt_n_paths_is_50(self) -> None:
        cfg, inp = _small()
        with patch(
            "dash_app.runner.optimize_household",
            return_value=(cfg, inp, {}),
        ) as m:
            run_scenario(
                cfg, inp,
                mode="four_strategies",
                seed=0,
                objective="cvar",
            )
        assert m.call_args.kwargs["n_paths"] == 50, (
            "stochastic objectives should default to 50 inner-MC "
            "paths for Dash interactivity; got "
            f"{m.call_args.kwargs['n_paths']}"
        )

    def test_explicit_opt_n_paths_overrides_default(self) -> None:
        cfg, inp = _small()
        with patch(
            "dash_app.runner.optimize_household",
            return_value=(cfg, inp, {}),
        ) as m:
            run_scenario(
                cfg, inp,
                mode="four_strategies",
                seed=0,
                objective="cvar",
                opt_n_paths=200,
            )
        assert m.call_args.kwargs["n_paths"] == 200

    def test_runner_n_paths_does_not_leak_into_optimizer(self) -> None:
        """The runner's user-facing `n_paths` argument controls the
        post-optimization fan-chart MC; it must NOT get passed
        through to ``optimize_household``'s inner MC (where it would
        spike the fitness-evaluation cost ~4× and re-introduce the
        Dash hang).
        """
        cfg, inp = _small()
        with patch(
            "dash_app.runner.optimize_household",
            return_value=(cfg, inp, {}),
        ) as m_opt:
            run_scenario(
                cfg, inp,
                mode="four_strategies",
                seed=0,
                objective="cvar",
                n_paths=2_000,  # Big — would hang if it leaked
            )
        # Optimizer's inner MC should still default to 50 even
        # though the user requested a 2,000-path post-MC.
        assert m_opt.call_args.kwargs["n_paths"] == 50


class TestDashLayoutControls:
    def test_layout_exposes_three_optimizer_controls(self) -> None:
        # Smoke check: the run-controls card now contains the three
        # new IDs. We don't render the whole layout, just confirm
        # the IDs are present in the component tree so the
        # callback wiring on the app side has something to bind to.
        from dash_app.layout import build_layout

        with _silence_dash_deprecations():
            layout = build_layout({})

        ids: list[str] = []

        def _walk(component) -> None:
            cid = getattr(component, "id", None)
            if isinstance(cid, str):
                ids.append(cid)
            children = getattr(component, "children", None)
            if children is None:
                return
            if isinstance(children, (list, tuple)):
                for c in children:
                    _walk(c)
            else:
                _walk(children)

        _walk(layout)

        for required in ("opt-objective", "opt-maxiter", "opt-popsize"):
            assert required in ids, (
                f"layout missing optimizer control id={required!r}"
            )
