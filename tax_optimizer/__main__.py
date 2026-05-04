"""CLI entrypoint: `python -m tax_optimizer`.

Default behavior runs the deterministic 4-strategy comparison + tornado
+ recommendations, mirroring the original `tax_optimizer.py` script.
Add `--monte-carlo N` to run N stochastic paths instead of (or in
addition to) the deterministic point estimate. Add `--widow YEARS` to
inject a death-of-spouse-A stress event N years into retirement, and
`--regime` to switch the tax regime.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

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


def _strategy_results(base_cfg: Config, inputs: Inputs) -> dict:
    s0 = base_cfg
    s1 = replace(base_cfg, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0)
    s2 = replace(base_cfg, roth_conversion_target_bracket=0.22)
    s3_cfg, _x = optimize_s3(base_cfg, inputs, objective="terminal")

    out: dict = {}
    for name, cfg in [
        ("S0_baseline", s0),
        ("S1_all_roth_401k", s1),
        ("S2_bracket_fill_22", s2),
        ("S3_optimized", s3_cfg),
    ]:
        df = simulate(cfg, inputs)
        out[name] = (cfg, df, summarize(df))
    return out


def _cfg_for_args(args: argparse.Namespace) -> Config:
    cfg = Config()

    if args.regime == "pre_tcja":
        cfg = replace(cfg, tax_regime=PRE_TCJA_2017)
    elif args.regime == "sunset":
        cfg = replace(cfg, tax_regime=SUNSET_2026)

    if args.regime_change_year is not None:
        target = TCJA_EXTENDED if args.regime == "sunset" else SUNSET_2026
        cfg = replace(
            cfg,
            regime_change_year_offset=args.regime_change_year,
            regime_change_target=target,
        )

    if args.widow:
        cfg = replace(cfg, mortality=Mortality(year_of_death_a=args.widow))

    if args.spending == "smile":
        cfg = replace(
            cfg,
            spending=SpendingProfile.retirement_smile(
                base_spending=cfg.annual_expenses_today, inflation=cfg.inflation
            ),
        )

    if args.market == "lognormal":
        cfg = replace(cfg, market=LognormalModel())
    elif args.market == "bootstrap":
        cfg = replace(cfg, market=BootstrapModel())
    else:
        cfg = replace(cfg, market=DeterministicModel(
            equity=cfg.nominal_growth_rate, bond=cfg.nominal_growth_rate
        ))

    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Retirement tax-strategy optimizer.")
    p.add_argument(
        "--regime",
        choices=["tcja", "pre_tcja", "sunset"],
        default="tcja",
        help="Active tax regime (default: tcja).",
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
        default="flat",
        help="Spending profile (default: flat).",
    )
    p.add_argument(
        "--market",
        choices=["deterministic", "lognormal", "bootstrap"],
        default="deterministic",
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
    args = p.parse_args()

    cfg = _cfg_for_args(args)
    inputs = Inputs()

    print(f"=== Tax optimizer ({cfg.tax_regime.name}) ===")
    print(
        f"Ages {cfg.spouse_a_age_start}/{cfg.spouse_b_age_start}, "
        f"retire {cfg.spouse_a_retire_age}/{cfg.spouse_b_retire_age}, "
        f"horizon {cfg.horizon_age}"
    )
    if args.widow:
        print(f"Stress test: Spouse A dies year +{args.widow}; survivor files Single.")
    if args.spending == "smile":
        print("Spending profile: retirement smile (go-go / slow-go / no-go + LTC shock)")
    if args.market != "deterministic":
        print(f"Market model: {args.market}")
    print()

    if args.monte_carlo > 0:
        mc = simulate_paths(
            cfg, inputs, n_paths=args.monte_carlo, seed=args.seed, keep_paths=False
        )
        print("=== Monte Carlo summary ===")
        for k, v in mc.summary().items():
            print(f"  {k:<26s} {v:,.4f}" if isinstance(v, float) else f"  {k:<26s} {v}")
        print()

    if args.mc_objective != "terminal" and args.monte_carlo > 0:
        print(f"=== Optimizer (objective={args.mc_objective}, paths={args.monte_carlo}) ===")
        opt_cfg, x_opt = optimize_s3(
            cfg,
            inputs,
            objective=args.mc_objective,
            n_paths=args.monte_carlo,
            seed=args.seed,
        )
        print(
            f"  spouse_a_roth_401k_pct = {opt_cfg.spouse_a_roth_401k_pct:.2f}\n"
            f"  spouse_b_roth_401k_pct = {opt_cfg.spouse_b_roth_401k_pct:.2f}\n"
            f"  roth_conversion_target = {opt_cfg.roth_conversion_target_bracket:.0%}\n"
        )
        return

    results = _strategy_results(cfg, inputs)
    print("=== Strategy comparison (deterministic) ===")
    print(f"{'strategy':<22}  {'tax_npv':>14}  {'irmaa_npv':>12}  {'terminal_atx':>16}")
    for name, (_c, _df, summ) in results.items():
        print(
            f"{name:<22}  {summ['lifetime_tax_npv']:>14,.0f}  "
            f"{summ['lifetime_irmaa_npv']:>12,.0f}  "
            f"{summ['terminal_after_tax']:>16,.0f}"
        )
    print()
    print(render_takeaways(results, cfg))
    print()
    sens, base_terminal = tornado_sensitivity(cfg, inputs)
    print(render_actions(results, sens, cfg, base_terminal))


if __name__ == "__main__":
    main()
