"""CLI entrypoint: `python -m tax_optimizer`.

Default behavior runs the deterministic 4-strategy comparison plus the
tornado sensitivity sweep and emits the same action plan that the
standalone notebook renders (household snapshot, recommended levers,
expected outcomes, top sensitivities, year-by-year timeline, year-by-
year withdrawal & conversion table, hygiene, caveats). Pass
`--monte-carlo N` to also include a Monte-Carlo risk section.

Output goes to the terminal pretty-printed via Rich when stdout is a
TTY, and as raw markdown when stdout is piped/redirected (so you can
still feed it to `glow`, `pandoc`, etc.). For a saved deliverable use
`--report PATH.html` or `--report PATH.pdf` -- only those two file
formats are supported, since they're what humans actually want to
read. PDF rendering requires `pip install 'tax-optimizer[pdf]'` plus
the WeasyPrint system libraries (pango/cairo). Pair with
`--report-archive` to also drop a timestamped HTML copy under
`reports/`. `--no-report` falls back to the older terse plain-text
output.

The CLI also supports three layered ways to override the built-in
defaults (lowest precedence first):

  1. `--scenario PATH`   — load a JSON file with `config` and/or `inputs`
                           sections (any field omitted keeps its default).
  2. The high-level flags `--regime`, `--market`, `--spending`, `--widow`,
     `--regime-change-year` apply on top of the scenario file.
  3. `--set DOTTED.PATH=VALUE` (repeatable) — surgical overrides parsed
     as JSON literals, e.g. `--set config.horizon_age=95`,
     `--set inputs.starting.hsa=25000`,
     `--set config.market='{"kind":"lognormal","equity_mu":0.07,"equity_sigma":0.18}'`.

Use `--print-defaults` to dump the current effective scenario (after
all flags are applied) as JSON, so you can save it as a starting point
for a custom file:

    tax-optimizer --print-defaults > my_plan.json

Add `--monte-carlo N` to run N stochastic paths instead of (or in
addition to) the deterministic point estimate. Add `--widow YEARS` to
inject a death-of-spouse-A stress event N years into retirement, and
`--regime` to switch the tax regime.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from . import (
    BootstrapModel,
    Config,
    DeterministicModel,
    Inputs,
    LognormalModel,
    Mortality,
    PRE_TCJA_2017,
    SUNSET_2026,
    SpendingProfile,
    TCJA_EXTENDED,
    optimize_s3,
    render_actions,
    render_takeaways,
    simulate,
    simulate_paths,
    summarize,
    tornado_sensitivity,
)
from .market import (
    CMA_PRESETS,
    HistoricalSequenceModel,
    MarketModel,
    lognormal_from_cma,
)
from .render import detect_format, render_terminal, write_report
from .report import build_action_report, cross_model_check
from .results import StrategyResult
from .scenario import (
    ScenarioError,
    apply_scenario,
    apply_set_overrides,
    load_scenario_file,
    scenario_to_dict,
)


# Built-in market-model kinds the `--cross-model` flag understands. The
# labels are matched against `_MODEL_NOTES` in `report.py`, so they need
# to stay in sync with that dict to keep the "Note" column populated.
_BUILTIN_CROSS_MODELS: dict[str, tuple[str, type]] = {
    "lognormal": ("LognormalModel", LognormalModel),
    "bootstrap": ("BootstrapModel", BootstrapModel),
    "historical_sequence": ("HistoricalSequenceModel", HistoricalSequenceModel),
    # Convenience alias for the long-form name.
    "historical": ("HistoricalSequenceModel", HistoricalSequenceModel),
}


class CrossModelError(ValueError):
    """Raised when `--cross-model` arguments can't be resolved."""


def _parse_cross_model_arg(arg: str | None) -> list[tuple[str, MarketModel]] | None:
    """Translate the raw `--cross-model` CLI argument into a model list.

    The input shapes accepted on the command line:

    * Flag absent (``arg is None``)   → returns ``None`` (caller should
      skip cross-model rendering entirely).
    * Flag present, no value (``arg == ""``) → returns ``None`` so
      ``cross_model_check`` uses its built-in defaults
      (``BootstrapModel`` + ``HistoricalSequenceModel``).
    * Comma-separated list of names  → returns a fully-resolved list of
      ``(label, MarketModel)`` tuples. Recognized names:

      - ``"lognormal"`` / ``"bootstrap"`` / ``"historical_sequence"``
        (alias: ``"historical"``) — the built-in market kinds.
      - Any key in ``tax_optimizer.market.CMA_PRESETS`` — instantiates
        a Lognormal model from the named CMA preset.

    Raises ``CrossModelError`` with a clear human-readable message when
    a name doesn't resolve, so the CLI can surface it cleanly.
    """
    if arg is None:
        return None
    if not arg.strip():
        return None  # Flag passed with no value → use cross_model_check defaults.

    raw_names = [n.strip() for n in arg.split(",") if n.strip()]
    if not raw_names:
        return None

    resolved: list[tuple[str, MarketModel]] = []
    for raw in raw_names:
        name = raw.lower()
        if name in _BUILTIN_CROSS_MODELS:
            label, cls = _BUILTIN_CROSS_MODELS[name]
            resolved.append((label, cls()))
            continue
        if name in CMA_PRESETS:
            resolved.append((name, lognormal_from_cma(name)))
            continue
        known = sorted({*_BUILTIN_CROSS_MODELS, *CMA_PRESETS})
        raise CrossModelError(
            f"--cross-model: unknown model {raw!r}. "
            f"Known: {', '.join(known)}"
        )
    return resolved


def _strategy_results(
    base_cfg: Config, base_inputs: Inputs
) -> dict[str, StrategyResult]:
    """Build the four canonical strategies (baseline / all-Roth-401k /
    bracket-fill / optimizer) and return their `StrategyResult`s.

    Each strategy ends up with its own `(cfg, inputs)` pair so the
    report renderer can show "current vs. recommended" cleanly.
    """
    s0 = (base_cfg, base_inputs)
    s1 = (
        base_cfg,
        replace(
            base_inputs, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0
        ),
    )
    s2 = (replace(base_cfg, roth_conversion_target_bracket=0.22), base_inputs)
    s3_cfg, s3_inputs, _x = optimize_s3(base_cfg, base_inputs, objective="terminal")
    s3 = (s3_cfg, s3_inputs)

    out: dict[str, StrategyResult] = {}
    for name, (cfg, inputs) in [
        ("S0_baseline", s0),
        ("S1_all_roth_401k", s1),
        ("S2_bracket_fill_22", s2),
        ("S3_optimized", s3),
    ]:
        df = simulate(cfg, inputs)
        out[name] = StrategyResult(
            cfg=cfg,
            inputs=inputs,
            df=df,
            summary=summarize(
                df,
                heir_marginal_rate=cfg.heir_marginal_rate,
                starting_balances=inputs.starting,
                inflation=cfg.inflation,
                retire_age=inputs.spouse_a_retire_age,
            ),
        )
    return out


def _apply_high_level_flags(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply the explicit high-level CLI flags on top of the given cfg.

    Only flags that the user actually supplied (i.e. not at their argparse
    default sentinel) take effect, so a `--scenario` file's settings are
    only overridden when the user passes the flag explicitly.
    """
    if args.regime is not None:
        if args.regime == "tcja":
            cfg = replace(cfg, tax_regime=TCJA_EXTENDED)
        elif args.regime == "pre_tcja":
            cfg = replace(cfg, tax_regime=PRE_TCJA_2017)
        elif args.regime == "sunset":
            cfg = replace(cfg, tax_regime=SUNSET_2026)

    if args.regime_change_year is not None:
        # Default flip target: opposite of current regime.
        target = TCJA_EXTENDED if cfg.tax_regime is SUNSET_2026 else SUNSET_2026
        cfg = replace(
            cfg,
            regime_change_year_offset=args.regime_change_year,
            regime_change_target=target,
        )

    if args.widow is not None:
        cfg = replace(cfg, mortality=Mortality(year_of_death_a=args.widow))

    if args.spending is not None:
        if args.spending == "smile":
            cfg = replace(
                cfg,
                spending=SpendingProfile.retirement_smile(
                    base_spending=cfg.annual_expenses_today, inflation=cfg.inflation
                ),
            )
        else:  # flat
            cfg = replace(cfg, spending=None)

    if args.market is not None:
        if args.market == "lognormal":
            cfg = replace(cfg, market=LognormalModel())
        elif args.market == "bootstrap":
            cfg = replace(cfg, market=BootstrapModel())
        else:  # deterministic
            cfg = replace(
                cfg,
                market=DeterministicModel(
                    equity=cfg.nominal_growth_rate, bond=cfg.nominal_growth_rate
                ),
            )

    return cfg


def _build_cfg_inputs(args: argparse.Namespace) -> tuple[Config, Inputs]:
    cfg, inputs = Config(), Inputs()

    if args.scenario:
        scenario = load_scenario_file(args.scenario)
        cfg, inputs = apply_scenario(cfg, inputs, scenario)

    cfg = _apply_high_level_flags(cfg, args)

    if args.set:
        cfg, inputs = apply_set_overrides(cfg, inputs, args.set)

    return cfg, inputs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tax-optimizer",
        description="Retirement tax-strategy optimizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tax-optimizer                              # pretty terminal report\n"
            "  tax-optimizer --report action_report.html  # styled HTML file\n"
            "  tax-optimizer --report plan.pdf            # PDF (needs [pdf] extra)\n"
            "  tax-optimizer --report-archive             # HTML archive under reports/\n"
            "  tax-optimizer --scenario my_plan.json --monte-carlo 500 --market lognormal\n"
            "  tax-optimizer --print-defaults > my_plan.json\n"
            "  tax-optimizer --set config.horizon_age=95 --set inputs.starting.hsa=25000\n"
            "  tax-optimizer --monte-carlo 1000 --cross-model                       # defaults\n"
            "  tax-optimizer --monte-carlo 1000 --cross-model bootstrap,vanguard_2025  # custom\n"
            "  tax-optimizer --no-report                  # legacy short text output\n"
            "  tax-optimizer | glow -                     # raw markdown when piped\n"
        ),
    )
    p.add_argument(
        "--scenario",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON file with `config` / `inputs` overrides. See README.",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="DOTTED.PATH=VALUE",
        help=(
            "Override a single field. Repeatable. Value is parsed as a JSON "
            "literal (numbers, true/false/null, quoted strings, lists). "
            "Bare strings are accepted unquoted."
        ),
    )
    p.add_argument(
        "--print-defaults",
        action="store_true",
        help=(
            "Dump the effective scenario (after --scenario/--set/flags) as "
            "JSON and exit. Useful as a template for custom scenario files."
        ),
    )

    p.add_argument(
        "--regime",
        choices=["tcja", "pre_tcja", "sunset"],
        default=None,
        help="Active tax regime (default: tcja unless set in --scenario).",
    )
    p.add_argument(
        "--regime-change-year",
        type=int,
        default=None,
        help="Year offset at which to swap regimes (mid-sim regime change).",
    )
    p.add_argument(
        "--widow",
        type=int,
        default=None,
        help="Stress test: year offset at which Spouse A dies.",
    )
    p.add_argument(
        "--spending",
        choices=["flat", "smile"],
        default=None,
        help="Spending profile (default: flat unless set in --scenario).",
    )
    p.add_argument(
        "--market",
        choices=["deterministic", "lognormal", "bootstrap"],
        default=None,
        help="Market return model.",
    )
    p.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        help="If >0, run a Monte Carlo with this many paths.",
    )
    p.add_argument(
        "--mc-objective",
        choices=["terminal", "cvar", "p_success"],
        default="terminal",
        help="Optimizer objective (Monte Carlo modes use n_paths from --monte-carlo).",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Master RNG seed for Monte Carlo."
    )
    p.add_argument(
        "--cross-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELS",
        help=(
            "Add a 'Cross-model robustness check' sub-section to §4 of the "
            "action report. Re-runs the winning plan under alternative market "
            "models. Pass with no value for the defaults (bootstrap + "
            "historical_sequence) or with a comma-separated list of: "
            "lognormal, bootstrap, historical_sequence (alias historical), "
            f"{', '.join(sorted(CMA_PRESETS))}. Requires --monte-carlo > 0."
        ),
    )
    p.add_argument(
        "--cross-model-paths",
        type=int,
        default=200,
        help=(
            "Number of Monte Carlo paths per alternative model when "
            "--cross-model is set (default: 200)."
        ),
    )
    p.add_argument(
        "--year-table-scope",
        choices=["full", "retirement"],
        default="full",
        help=(
            "Scope of the §7 year-by-year withdrawal & conversion table. "
            "'full' (default) shows every simulated year including "
            "working years with their AGI/federal tax columns; "
            "'retirement' shows retirement years only (the legacy "
            "v1-v6 compact view)."
        ),
    )

    p.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write the action plan to PATH (must end in .html or .pdf). "
            "If PATH is an existing directory, the file is named "
            "action_report.html inside it. When this flag is passed, "
            "stdout is suppressed unless --also-stdout is given."
        ),
    )
    p.add_argument(
        "--report-archive",
        action="store_true",
        help=(
            "In addition to the regular output, drop a timestamped HTML "
            "copy under ./reports/action_report_YYYY-MM-DD_HHMMSS.html."
        ),
    )
    p.add_argument(
        "--also-stdout",
        action="store_true",
        help=(
            "When --report PATH writes to a file, also print the report "
            "to stdout (pretty if a TTY, raw markdown if piped)."
        ),
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help=(
            "Skip the action plan and emit the older short text summary "
            "instead (strategy table + render_takeaways + render_actions)."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress stdout output of the report (only meaningful with "
            "--report PATH or --report-archive)."
        ),
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    try:
        cfg, inputs = _build_cfg_inputs(args)
    except ScenarioError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.print_defaults:
        json.dump(scenario_to_dict(cfg, inputs), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if args.mc_objective != "terminal" and args.monte_carlo > 0:
        _run_objective_optimizer(cfg, inputs, args)
        return

    mc = None
    if args.monte_carlo > 0:
        mc = simulate_paths(
            cfg, inputs, n_paths=args.monte_carlo, seed=args.seed, keep_paths=False
        )

    extra_mc = None
    if args.cross_model is not None:
        if mc is None:
            print(
                "error: --cross-model requires --monte-carlo > 0 "
                "(the alternative models anchor against the main MC result).",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            cross_models = _parse_cross_model_arg(args.cross_model)
        except CrossModelError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        extra_mc = cross_model_check(
            cfg,
            inputs,
            n_paths=args.cross_model_paths,
            seed=args.seed if args.seed else 42,
            models=cross_models,
        )

    results = _strategy_results(cfg, inputs)
    sens, base_terminal = tornado_sensitivity(cfg, inputs)

    if args.no_report:
        _emit_legacy_text(cfg, inputs, args, results, sens, base_terminal, mc)
        return

    report_md = build_action_report(
        cfg=cfg,
        inputs=inputs,
        results=results,
        sens_df=sens,
        base_terminal=base_terminal,
        mc=mc,
        scenario_path=args.scenario,
        extra_mc=extra_mc,
        year_table_scope=args.year_table_scope,
    )
    _emit_report(report_md, args)


def _emit_report(report_md: str, args: argparse.Namespace) -> None:
    """Render the report: pretty stdout by default, file output if asked."""
    written: list[tuple[Path, str]] = []

    try:
        if args.report:
            target = _resolve_report_path(args.report)
            fmt = write_report(report_md, target)
            written.append((target, fmt))

        if args.report_archive:
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            archive = Path("reports") / f"action_report_{stamp}.html"
            fmt = write_report(report_md, archive)
            written.append((archive, fmt))
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    suppress_stdout = args.quiet or (bool(written) and not args.also_stdout)
    if not suppress_stdout:
        render_terminal(report_md)

    for path, fmt in written:
        print(f"wrote {path}  [{fmt}]", file=sys.stderr)


def _resolve_report_path(raw: str) -> Path:
    """Resolve `--report PATH`. Directories receive `action_report.html`."""
    path = Path(raw)
    if path.exists() and path.is_dir():
        return path / "action_report.html"
    detect_format(path)  # validate extension up front for a clean error
    return path


def _run_objective_optimizer(
    cfg: Config, inputs: Inputs, args: argparse.Namespace
) -> None:
    """`--mc-objective {cvar,p_success}` path: bespoke optimizer summary."""
    print(f"=== Tax optimizer ({cfg.tax_regime.name}) ===")
    print(
        f"Ages {inputs.spouse_a_age_start}/{inputs.spouse_b_age_start}, "
        f"retire {inputs.spouse_a_retire_age}/{inputs.spouse_b_retire_age}, "
        f"horizon {cfg.horizon_age}"
    )
    print(f"=== Optimizer (objective={args.mc_objective}, paths={args.monte_carlo}) ===")
    # `seed=` reseeds the differential-evolution sampler; `mc_seed=`
    # reseeds the Monte Carlo path draws that the objective scores
    # against. Both have to share `args.seed` for the CLI's
    # `--seed` flag to make MC objectives reproducible AND varied
    # across distinct seeds.
    opt_cfg, opt_inputs, _x = optimize_s3(
        cfg,
        inputs,
        objective=args.mc_objective,
        n_paths=args.monte_carlo,
        seed=args.seed,
        mc_seed=args.seed,
    )
    print(
        f"  spouse_a_roth_401k_pct = {opt_inputs.spouse_a_roth_401k_pct:.2f}\n"
        f"  spouse_b_roth_401k_pct = {opt_inputs.spouse_b_roth_401k_pct:.2f}\n"
        f"  roth_conversion_target = {opt_cfg.roth_conversion_target_bracket:.0%}\n"
    )


def _emit_legacy_text(
    cfg: Config,
    inputs: Inputs,
    args: argparse.Namespace,
    results: dict[str, StrategyResult],
    sens,
    base_terminal: float,
    mc,
) -> None:
    """The pre-report terse output, kept behind --no-report."""
    print(f"=== Tax optimizer ({cfg.tax_regime.name}) ===")
    print(
        f"Ages {inputs.spouse_a_age_start}/{inputs.spouse_b_age_start}, "
        f"retire {inputs.spouse_a_retire_age}/{inputs.spouse_b_retire_age}, "
        f"horizon {cfg.horizon_age}"
    )
    if args.scenario:
        print(f"Scenario file: {args.scenario}")
    if args.widow:
        print(f"Stress test: Spouse A dies year +{args.widow}; survivor files Single.")
    if args.spending == "smile":
        print("Spending profile: retirement smile (go-go / slow-go / no-go + LTC shock)")
    if args.market and args.market != "deterministic":
        print(f"Market model: {args.market}")
    print()

    if mc is not None:
        print("=== Monte Carlo summary ===")
        for k, v in mc.summary().items():
            print(f"  {k:<26s} {v:,.4f}" if isinstance(v, float) else f"  {k:<26s} {v}")
        print()

    print("=== Strategy comparison (deterministic) ===")
    print(f"{'strategy':<22}  {'tax_npv':>14}  {'irmaa_npv':>12}  {'terminal_atx':>16}")
    for name, r in results.items():
        summ = r.summary
        print(
            f"{name:<22}  {summ['lifetime_tax_npv']:>14,.0f}  "
            f"{summ['lifetime_irmaa_npv']:>12,.0f}  "
            f"{summ['terminal_after_tax']:>16,.0f}"
        )
    print()
    print(render_takeaways(results, cfg, inputs))
    print()
    print(render_actions(results, sens, cfg, base_terminal, inputs))


if __name__ == "__main__":
    main()
