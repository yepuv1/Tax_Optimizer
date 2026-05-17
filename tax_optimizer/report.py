"""Markdown action-plan report.

This is the same report the standalone notebook renders in section 9
("Recommended actions") -- ported here so the CLI can produce it by
default and so anyone using the Python API can call
`build_action_report(cfg, inputs, results, sens_df, base_terminal, mc=...)`
to get the same markdown string the notebook displays.

The report is intentionally self-contained markdown with no plots: it
prints fine to a terminal, renders cleanly on GitHub, and survives
copy-paste into email or Slack.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dataclasses import replace as _replace

from .config import Config
from .inputs import Inputs
from .market import (
    BootstrapModel,
    HistoricalSequenceModel,
    LognormalModel,
    MarketModel,
)
from .metrics import summarize, terminal_after_tax_nw
from .monte_carlo import MonteCarloResult, simulate_paths
from .results import StrategyResult
from .simulator import simulate


# ---------------------------------------------------------------------
# Cross-model robustness helper (R-D.1)
# ---------------------------------------------------------------------


_MODEL_NOTES: dict[str, str] = {
    "LognormalModel": "Parametric (μ, σ, ρ from cfg)",
    "BootstrapModel": "Historical-tail aware (block-resampled 1928–2023)",
    "HistoricalSequenceModel": "Preserves return sequences (~50 windows)",
    "DeterministicModel": "Single path, no uncertainty",
    # CMA-preset labels emitted by the CLI's `--cross-model` flag.
    "vanguard_2025": "Lognormal w/ Vanguard CMA 2025 (μ_eq=5.5%, ρ=+0.15)",
    "jpm_ltcma_2025": "Lognormal w/ JPM LTCMA 2025 (μ_eq=7.2%, ρ=+0.12)",
    "horizon_2025": "Lognormal w/ Horizon Survey 2025 (μ_eq=7.2%, ρ=+0.10)",
    "historical_1928_2023": "Lognormal calibrated to 1928–2023 history",
    "historical_1985_2023": "Lognormal calibrated to post-Volcker 1985–2023",
}


def cross_model_check(
    cfg: Config,
    inputs: Inputs,
    *,
    n_paths: int = 200,
    seed: int = 42,
    models: list[tuple[str, MarketModel]] | None = None,
) -> dict[str, MonteCarloResult]:
    """Re-score the *same* household plan under multiple market models.

    Returns a `{name: MonteCarloResult}` dict suitable for passing as
    `build_action_report(..., extra_mc=...)`. The plan stays the same;
    only the return-generating process changes. This protects against
    "my plan only looks safe because I picked a friendly market model"
    — if `BootstrapModel` drops P(success) from 100% to 85% relative to
    the parametric `LognormalModel`, the user wants to know.

    Default models: `BootstrapModel`, `HistoricalSequenceModel`. Both
    are non-parametric (historical-tail aware vs sequence-preserving),
    complementing the parametric `LognormalModel` that's typically
    already on `cfg.market` and passed as the main `mc=` argument.

    Pass `models=...` to override (e.g. include a custom CAPE-conditioned
    Lognormal, or skip HistoricalSequence on very long horizons where
    only ~3 windows would fit).

    Compute cost ≈ `len(models) × n_paths × horizon_years` simulations.
    With defaults that's ~12k single-year sims, ~2-4 seconds on typical
    hardware.
    """
    if models is None:
        models = [
            ("BootstrapModel", BootstrapModel()),
            ("HistoricalSequenceModel", HistoricalSequenceModel()),
        ]
    out: dict[str, MonteCarloResult] = {}
    for label, model in models:
        scoped_cfg = _replace(cfg, market=model)
        out[label] = simulate_paths(
            scoped_cfg,
            inputs,
            n_paths=n_paths,
            seed=seed,
            keep_paths=False,
        )
    return out


def _cross_model_table(
    mc: MonteCarloResult, extra_mc: dict[str, MonteCarloResult]
) -> list[str]:
    """Render the cross-model robustness sub-section for §4.

    The main `mc` result is listed first (tagged "current"); each
    entry in `extra_mc` becomes another row. P(success) below 90%
    in any row triggers a callout block — the rest of the report
    might say the plan is safe under the parametric model, but if
    a historical-tail resample disagrees, the reader should see it.
    """
    out = ["### Cross-model robustness check", ""]
    out.append(
        "_Re-runs the same plan under additional market models so you can "
        "see whether the conclusion depends on which return-generating "
        "process you trust. Each row uses the same household plan; only "
        "the market model changes._"
    )
    out.append("")
    out.append("| Market model | P(success) | Median terminal NW | CVaR(10%) | Note |")
    out.append("|---|---:|---:|---:|---|")

    main_name = type(mc.cfg.resolved_market()).__name__
    main_note = _MODEL_NOTES.get(main_name, "—")
    s_main = mc.summary()
    out.append(
        f"| `{main_name}` (current) | {s_main['prob_success']:.1%} | "
        f"${s_main['terminal_p50']:,.0f} | "
        f"${s_main['cvar_terminal_p10']:,.0f} | {main_note} |"
    )

    failing = s_main["prob_success"] < 0.90
    p_success_values = [s_main["prob_success"]]
    for label, extra in extra_mc.items():
        s = extra.summary()
        note = _MODEL_NOTES.get(label, "—")
        out.append(
            f"| `{label}` | {s['prob_success']:.1%} | "
            f"${s['terminal_p50']:,.0f} | "
            f"${s['cvar_terminal_p10']:,.0f} | {note} |"
        )
        if s["prob_success"] < 0.90:
            failing = True
        p_success_values.append(s["prob_success"])

    out.append("")
    spread = max(p_success_values) - min(p_success_values)
    if failing:
        out.append(
            "> ⚠️ **Model-risk flag:** at least one market model shows "
            "P(success) below 90%. The plan is sensitive to which return "
            "process you trust — consider a more conservative withdrawal "
            "strategy, a larger cash buffer, or stress-testing with "
            "`objective=\"cvar\"` instead of `terminal`."
        )
    elif spread >= 0.05:
        out.append(
            f"> _P(success) ranges {min(p_success_values):.0%}–"
            f"{max(p_success_values):.0%} across models — small enough "
            f"to be noise, but worth tracking if you tighten the plan._"
        )
    else:
        out.append(
            "> _All models agree the plan is robust (P(success) within "
            "5pp). The conclusion is not an artifact of the market-model "
            "assumption._"
        )
    out.append("")
    return out


def _winning_strategy(results: dict[str, StrategyResult]) -> str:
    return max(results, key=lambda n: results[n].summary["terminal_after_tax"])


def _peak_marginal_year(df: pd.DataFrame) -> dict[str, Any]:
    """Age and AGI of the year with the highest marginal rate."""
    idx = df["marginal"].idxmax()
    row = df.loc[idx]
    return {
        "age": int(row["spouse_a_age"]),
        "agi": float(row["agi"]),
        "marginal": float(row["marginal"]),
    }


def _conversion_window(inputs: Inputs, cfg: Config) -> tuple[int, int]:
    """Age range where Roth conversions are most useful: from the later-
    retiring spouse's retire age until RMDs hit."""
    start_age = max(inputs.spouse_a_retire_age, inputs.spouse_b_retire_age)
    end_age = cfg.rmd_start_age - 1
    return start_age, end_age


def _lever_changes(
    base_cfg: Config,
    base_inputs: Inputs,
    w_cfg: Config,
    w_inputs: Inputs,
) -> list[dict]:
    """Return one dict per decision-vector axis whose value differs
    between the user-supplied inputs and the optimizer-chosen plan.

    Used by the TL;DR section so the reader can immediately see *what
    the optimizer wants you to change*. Axes that match the baseline
    are omitted entirely (no noise).
    """
    out: list[dict] = []
    pct_pairs = [
        ("Spouse A Roth-401(k) share", "spouse_a_roth_401k_pct", base_inputs, w_inputs),
        ("Spouse B Roth-401(k) share", "spouse_b_roth_401k_pct", base_inputs, w_inputs),
        ("Spouse A mega-backdoor share", "spouse_a_after_tax_401k_pct", base_inputs, w_inputs),
        ("Spouse B mega-backdoor share", "spouse_b_after_tax_401k_pct", base_inputs, w_inputs),
        ("Roth conversion target bracket", "roth_conversion_target_bracket", base_cfg, w_cfg),
    ]
    for label, attr, base_obj, w_obj in pct_pairs:
        b = getattr(base_obj, attr, None)
        a = getattr(w_obj, attr, None)
        if b is None or a is None:
            continue
        if abs(float(a) - float(b)) < 1e-6:
            continue
        out.append({"label": label, "before": float(b), "after": float(a), "kind": "pct"})
    age_pairs = [
        ("Spouse A SS claim age", "start_age_a"),
        ("Spouse B SS claim age", "start_age_b"),
    ]
    for label, attr in age_pairs:
        b = getattr(base_inputs.ss, attr, None)
        a = getattr(w_inputs.ss, attr, None)
        if b is None or a is None or b == a:
            continue
        out.append({"label": label, "before": int(b), "after": int(a), "kind": "age"})
    return out


def _market_summary(cfg: Config) -> str:
    """Short human description of the market model the simulator will
    actually use. For the parametric lognormal model we also surface
    the equity/bond mu, sigma and correlation so an advisor can
    sanity-check the assumptions.

    Routes through `cfg.resolved_market()` so the default `Config()`
    (where `cfg.market is None`) renders the deterministic fallback
    rather than the literal string ``"NoneType"``.
    """
    market = cfg.resolved_market()
    name = type(market).__name__
    if isinstance(market, LognormalModel):
        return (
            f"{name} "
            f"(equity μ={market.equity_mu:.1%}/σ={market.equity_sigma:.1%}, "
            f"bond μ={market.bond_mu:.1%}/σ={market.bond_sigma:.1%}, "
            f"ρ={market.equity_bond_corr:.2f})"
        )
    return name


def _assumptions_block(cfg: Config, inputs: Inputs) -> list[str]:
    """Return the markdown lines for the "Assumptions driving this
    plan" subsection. Always renders the same shape so users learn
    where to look for each knob."""
    horizon = cfg.horizon_age - inputs.spouse_a_age_start + 1
    out = ["### Assumptions driving this plan", ""]
    out.append("| Assumption | Value |")
    out.append("|---|---|")
    out.append(
        f"| Heir marginal tax rate (pretax / HSA bequest haircut) "
        f"| {cfg.heir_marginal_rate:.0%} |"
    )
    out.append(f"| Inflation | {cfg.inflation:.1%}/yr |")
    out.append(
        f"| Nominal growth (deterministic baseline) | {cfg.nominal_growth_rate:.1%}/yr |"
    )
    out.append(f"| Market model | {_market_summary(cfg)} |")
    if cfg.regime_change_year_offset is not None and cfg.regime_change_target is not None:
        regime = (
            f"`{cfg.tax_regime.name}` → `{cfg.regime_change_target.name}` "
            f"in year {cfg.regime_change_year_offset}"
        )
    else:
        regime = f"`{cfg.tax_regime.name}` (no scheduled change)"
    out.append(f"| Federal tax regime | {regime} |")
    out.append(f"| State tax regime | `{cfg.state_regime.name}` |")
    out.append(
        f"| Mortality | Spouse A: year {cfg.mortality.year_of_death_a}, "
        f"Spouse B: year {cfg.mortality.year_of_death_b} (horizon = year {horizon}) |"
    )
    out.append(
        f"| Step-up in basis at first death | "
        f"{'on' if cfg.stepup_at_first_death else 'off'} |"
    )
    out.append(
        f"| ACA premium tax credit | "
        f"{'enabled' if cfg.aca_enabled else 'disabled'} |"
    )
    out.append(
        f"| Pre-65 healthcare (today's $) | ${cfg.health_pre65_today:,.0f}/yr |"
    )
    out.append(
        f"| Medicare base premium (today's $) | "
        f"${cfg.medicare_base_b_d_premium:,.0f}/yr |"
    )
    out.append(f"| IRMAA MAGI lookback | {cfg.irmaa_lookback_years} year(s) |")
    out.append("")
    return out


def _optimizer_rationale(
    cfg: Config,
    inputs: Inputs,
    w_cfg: Config,
    w_inputs: Inputs,
    w_df: pd.DataFrame,
    w_summary: dict,
) -> list[str]:
    """Render heuristic bullet points explaining *why* the winning plan
    looks the way it does.

    The optimizer is a black-box differential evolution &mdash; we can't read
    its mind. But we can recognise a handful of common cause-and-effect
    patterns from the recommended levers, the household's starting
    balances, the configured assumptions, and the year-by-year frame.
    Each pattern below is checked in priority order; we render the
    first 3 that fire so the reader gets a focused narrative, not a
    laundry list.
    """
    out: list[str] = []
    bullets: list[str] = []

    heir_rate = float(cfg.heir_marginal_rate)
    peak_marginal = float(w_summary.get("peak_marginal", 0.0))

    retire_age = max(inputs.spouse_a_retire_age, inputs.spouse_b_retire_age)
    retire_rows = w_df[w_df["spouse_a_age"] >= retire_age]
    avg_retire_marginal = (
        float(retire_rows["marginal"].mean()) if len(retire_rows) else 0.0
    )

    total_pretax = (
        inputs.starting.spouse_a_pretax_401k + inputs.starting.spouse_b_pretax_401k
        + inputs.starting.spouse_a_pretax_ira + inputs.starting.spouse_b_pretax_ira
    )
    total_roth = inputs.starting.spouse_a_roth_ira + inputs.starting.spouse_b_roth_ira
    pretax_to_roth_ratio = (
        total_pretax / total_roth if total_roth > 1 else float("inf")
    )

    chose_conversions = float(w_cfg.roth_conversion_target_bracket) > 0
    has_sunset = (
        cfg.regime_change_year_offset is not None
        and cfg.regime_change_target is not None
    )

    # Pattern A — heir-rate framing for the Roth-conversion stance.
    if chose_conversions and heir_rate >= avg_retire_marginal - 0.005:
        bullets.append(
            f"**Heir-rate framing:** your heir marginal rate ({heir_rate:.0%}) "
            f"meets-or-exceeds your average retirement marginal "
            f"({avg_retire_marginal:.0%}). Each pretax dollar you do "
            f"*not* convert will be taxed at {heir_rate:.0%} on the "
            f"bequest, so converting now (paying ≤ {avg_retire_marginal:.0%}) "
            f"locks in a cheaper rate. Hence the "
            f"{w_cfg.roth_conversion_target_bracket:.0%} conversion target."
        )
    elif not chose_conversions:
        bullets.append(
            f"**No conversions chosen:** your average retirement marginal "
            f"({avg_retire_marginal:.0%}) is already at-or-above your heir "
            f"rate ({heir_rate:.0%}), so a conversion would cost more tax "
            f"now than it would save on the bequest. The optimizer keeps "
            f"the dollars pretax."
        )

    # Pattern B — sunset front-loading.
    if has_sunset and chose_conversions:
        offset = cfg.regime_change_year_offset
        cal = cfg.start_year + offset
        bullets.append(
            f"**Sunset front-loading:** the federal regime steps to "
            f"`{cfg.regime_change_target.name}` in year {offset} "
            f"(calendar {cal}); brackets compress and rates rise. "
            f"Concentrating Roth conversions *before* the boundary locks "
            f"in current-law rates."
        )

    # Pattern C — pretax-heavy starting balance.
    if pretax_to_roth_ratio > 5:
        if pretax_to_roth_ratio == float("inf"):
            ratio_text = "≫"
        else:
            ratio_text = f"{pretax_to_roth_ratio:.0f}×"
        bullets.append(
            f"**Bucket imbalance:** your pretax balance is {ratio_text} "
            f"your Roth balance (${total_pretax:,.0f} vs "
            f"${total_roth:,.0f}). The optimizer raises Roth-401(k) "
            f"share and conversions to rebalance — left unchecked, RMDs "
            f"on a {ratio_text} pretax pile push you into IRMAA and the "
            f"32-37% bracket every year after age {cfg.rmd_start_age}."
        )

    # Pattern D — mega-backdoor activity.
    mega_total = float(
        w_df.get("mega_backdoor_a", pd.Series([0])).sum()
        + w_df.get("mega_backdoor_b", pd.Series([0])).sum()
    )
    if mega_total > 0:
        years_active = int(
            ((w_df.get("mega_backdoor_a", 0) > 0) | (w_df.get("mega_backdoor_b", 0) > 0)).sum()
        )
        bullets.append(
            f"**Mega-backdoor lever pulled:** the plan routes "
            f"${mega_total:,.0f} through the after-tax 401(k) → in-plan "
            f"Roth conversion path over {years_active} year(s). This is "
            f"Roth headroom *above* the elective-deferral cap, only "
            f"available if your plan supports in-service after-tax "
            f"contributions and conversions."
        )

    # Pattern E — peak marginal sticker shock.
    peak_year_age = int(w_df.iloc[w_df["marginal"].idxmax()]["spouse_a_age"])
    if peak_marginal >= 0.32 and chose_conversions:
        bullets.append(
            f"**Peak marginal flag:** the plan hits {peak_marginal:.0%} "
            f"at age {peak_year_age}. That's the bracket the optimizer "
            f"is willing to pay during conversions because the heir "
            f"alternative is worse — but check whether you can spread "
            f"conversions over more years to stay below this rate."
        )

    if not bullets:
        bullets.append(
            "_The optimizer's choice closely matches your inputs; no "
            "single pattern dominates the rationale._"
        )

    out.append("### Why the optimizer chose this")
    out.append("")
    for b in bullets[:3]:
        out.append(f"- {b}")
    out.append("")
    return out


def _heir_rate_sensitivity(
    w_cfg: Config, w_df: pd.DataFrame
) -> list[str]:
    """Show how the score changes if the heir-tax assumption is wrong.

    `cfg.heir_marginal_rate` is the single most influential knob in the
    objective (see the `terminal_after_tax_nw` docstring). The plan
    itself doesn't change here &mdash; we just re-score the same year-by-year
    frame at three alternative heir rates so the reader sees the
    sensitivity quantified before adopting the recommendation.
    """
    rates = [0.10, 0.22, 0.32, 0.37]
    current = float(w_cfg.heir_marginal_rate)
    if not any(abs(r - current) < 1e-3 for r in rates):
        rates.append(current)
        rates = sorted(set(rates))
    base = terminal_after_tax_nw(w_df, heir_marginal_rate=current)

    out = ["### Sensitivity to the heir-tax assumption", ""]
    out.append(
        "_The optimizer's verdict depends on how aggressively you "
        "discount terminal pretax dollars by the heir's eventual "
        "tax bill (see the `Assumptions driving this plan` block). "
        "The plan stays the same in each row below — we just re-score "
        "it at a different heir rate, so you can see how robust the "
        "answer is to your view of future tax rates._"
    )
    out.append("")
    out.append("| Heir marginal rate | Terminal after-tax NW | Δ vs current |")
    out.append("|---:|---:|---:|")
    for r in rates:
        nw = terminal_after_tax_nw(w_df, heir_marginal_rate=r)
        delta = nw - base
        marker = "  ← current assumption" if abs(r - current) < 1e-3 else ""
        out.append(
            f"| **{r:.0%}**{marker} | ${nw:,.0f} | "
            f"{'$' + format(delta, '+,.0f') if abs(delta) >= 1 else '—'} |"
        )
    out.append("")
    return out


def _this_year_actions(
    cfg: Config, inputs: Inputs, w_df: pd.DataFrame
) -> list[str]:
    """Render an immediately-actionable list of year-1 dollar amounts.

    The big-picture timeline (multi-decade phases) is useful for orientation
    but doesn't tell the reader what to do *this week*. This block extracts
    year-1 contributions, conversions, and the expected tax bill from the
    winning plan's first simulated year so the reader has concrete numbers
    to set up payroll deductions and HSA / brokerage transfers.
    """
    y0 = w_df.iloc[0]
    calendar_year = int(y0["year"])
    out = [f"### This year's concrete actions (calendar year {calendar_year})", ""]
    out.append(
        "_Year-1 dollar amounts from the optimized plan. Use these to set "
        "up payroll deductions and HSA / brokerage transfers immediately._"
    )
    out.append("")

    elec_a = float(y0.get("elective_deferral_a", 0) or 0)
    elec_b = float(y0.get("elective_deferral_b", 0) or 0)
    if elec_a + elec_b > 0:
        out.append(
            f"- **401(k) employee deferral:** Spouse A ${elec_a:,.0f} "
            f"({inputs.spouse_a_roth_401k_pct:.0%} Roth) + "
            f"Spouse B ${elec_b:,.0f} "
            f"({inputs.spouse_b_roth_401k_pct:.0%} Roth) "
            f"= **${elec_a + elec_b:,.0f} total**"
        )

    match_a = float(y0.get("employer_match_a", 0) or 0)
    match_b = float(y0.get("employer_match_b", 0) or 0)
    if match_a + match_b > 0:
        out.append(
            f"- **Employer 401(k) match (free money):** "
            f"Spouse A ${match_a:,.0f} + Spouse B ${match_b:,.0f} "
            f"= **${match_a + match_b:,.0f} total**"
        )

    mega_a = float(y0.get("mega_backdoor_a", 0) or 0)
    mega_b = float(y0.get("mega_backdoor_b", 0) or 0)
    if mega_a + mega_b > 0:
        out.append(
            f"- **Mega-backdoor Roth (after-tax 401(k) → in-plan conversion):** "
            f"Spouse A ${mega_a:,.0f} + Spouse B ${mega_b:,.0f} "
            f"= **${mega_a + mega_b:,.0f} total**"
        )

    bd_a = float(y0.get("ira_backdoor_a", 0) or 0)
    bd_b = float(y0.get("ira_backdoor_b", 0) or 0)
    if bd_a + bd_b > 0:
        out.append(
            f"- **Backdoor Roth IRA:** Spouse A ${bd_a:,.0f} + "
            f"Spouse B ${bd_b:,.0f} = **${bd_a + bd_b:,.0f} total**"
        )

    hsa = float(y0.get("hsa_contrib", 0) or 0)
    if hsa > 0:
        out.append(f"- **HSA:** ${hsa:,.0f}")

    conv = float(y0.get("roth_conversion", 0) or 0)
    retire_age = max(inputs.spouse_a_retire_age, inputs.spouse_b_retire_age)
    in_retirement = int(y0["spouse_a_age"]) >= retire_age
    if conv > 0:
        out.append(
            f"- **Roth conversion (pretax → Roth):** ${conv:,.0f} — "
            f"pay the conversion tax from taxable, not from the converted amount"
        )
    elif not in_retirement:
        out.append(
            "- **Roth conversion:** none this year — wages still high; "
            "conversion window opens after retirement"
        )
    else:
        out.append(
            "- **Roth conversion:** none this year — household is already "
            "in a bracket at or above the conversion target"
        )

    rmd = float(y0.get("rmd", 0) or 0)
    if rmd > 0:
        out.append(
            f"- **RMD (mandatory pretax withdrawal):** ${rmd:,.0f} — "
            f"take by Dec 31 to avoid the 25% excess-accumulation penalty"
        )

    fed = float(y0.get("federal_tax", 0) or 0)
    state = float(y0.get("state_tax", 0) or 0)
    fica = float(y0.get("fica", 0) or 0)
    out.append("")
    out.append(
        f"**Expected year-1 tax bill:** federal ${fed:,.0f} + "
        f"state ${state:,.0f} + FICA ${fica:,.0f} = "
        f"**${fed + state + fica:,.0f}**"
    )
    out.append("")
    return out


def _lifetime_events(
    cfg: Config, inputs: Inputs, w_df: pd.DataFrame
) -> list[str]:
    """Render any one-time / pay-period income events the simulator
    recorded in the year-by-year DataFrame.

    The action-plan markdown previously walked the optimized strategy's
    contributions, RMDs, and Roth conversions but ignored the
    pension-lump-sum / annuity-payout / annuity-lump / §72(t) early-
    distribution events even though the simulator tracks them in
    explicit columns (`pension_lump_sum`, `annuity_lump_sum`,
    `annuity_payment`, `annuity_taxable`, `early_distribution_penalty`).
    We surface them here so the reader sees:

      * which year the pension lump fires (cash vs rollover) and how
        much of the cash mode is hit by the 10% IRC §72(t) penalty,
      * which year the deferred annuity starts paying (and any annuity
        lump in cash or rollover mode), and
      * any year where the early-distribution penalty exceeds zero
        (catches partial-year mid-59½ cases too).

    Returns an empty list when none of those events fire — most
    households see nothing here, and that's correct behavior.
    """
    if w_df.empty:
        return []

    out: list[str] = []
    rows: list[str] = []

    pen_lump_rows = w_df[w_df.get("pension_lump_sum", 0) > 0]
    for _, row in pen_lump_rows.iterrows():
        cal_year = int(row["year"])
        a_age = int(row["spouse_a_age"])
        amt = float(row["pension_lump_sum"])
        event = str(row.get("pension_lump_sum_event", "") or "")
        penalty = float(row.get("early_distribution_penalty", 0) or 0)
        if event == "rollover_pretax":
            note = "direct rollover into pretax IRA — no current-year tax"
        elif event == "cash":
            if penalty > 0:
                note = (
                    f"taxable lump-sum distribution; **§72(t) 10% "
                    f"surtax applies** (${penalty:,.0f})"
                )
            else:
                note = "taxable lump-sum distribution (post-59½, no penalty)"
        else:
            note = "pension lump-sum event"
        rows.append(
            f"| {cal_year} | A age {a_age} | Pension lump-sum "
            f"({event or 'event'}) | ${amt:,.0f} | {note} |"
        )

    ann_lump_rows = w_df[w_df.get("annuity_lump_sum", 0) > 0]
    for _, row in ann_lump_rows.iterrows():
        cal_year = int(row["year"])
        a_age = int(row["spouse_a_age"])
        amt = float(row["annuity_lump_sum"])
        event = str(row.get("annuity_lump_sum_event", "") or "")
        penalty = float(row.get("early_distribution_penalty", 0) or 0)
        if event == "rollover_pretax":
            note = "direct rollover into pretax IRA — no current-year tax"
        elif event == "cash":
            if penalty > 0:
                note = (
                    f"non-qualified surrender; **§72(q) 10% surtax "
                    f"applies on the gain** (${penalty:,.0f})"
                )
            else:
                note = "annuity surrender (post-59½, no penalty)"
        else:
            note = "annuity lump-sum event"
        rows.append(
            f"| {cal_year} | A age {a_age} | Annuity lump-sum "
            f"({event or 'event'}) | ${amt:,.0f} | {note} |"
        )

    payment_col = w_df.get("annuity_payment")
    if payment_col is not None:
        positive = (payment_col.fillna(0) > 0)
        if positive.any():
            first_idx = positive.idxmax()
            first_row = w_df.loc[first_idx]
            cal_year = int(first_row["year"])
            a_age = int(first_row["spouse_a_age"])
            amt = float(first_row["annuity_payment"])
            taxable = float(first_row.get("annuity_taxable", 0) or 0)
            tax_free = float(first_row.get("annuity_tax_free", 0) or 0)
            if tax_free > 0:
                note = (
                    f"first scheduled payment; ${taxable:,.0f} taxable "
                    f"+ ${tax_free:,.0f} tax-free basis return"
                )
            else:
                note = "first scheduled payment (fully taxable)"
            rows.append(
                f"| {cal_year} | A age {a_age} | Annuity payments begin "
                f"| ${amt:,.0f} | {note} |"
            )

    penalty_col = w_df.get("early_distribution_penalty")
    if penalty_col is not None:
        for idx, val in penalty_col.items():
            if not (val and val > 0):
                continue
            row = w_df.loc[idx]
            # Skip rows already explained by the pension-lump or
            # annuity-lump branch above (their note covers the
            # penalty); only surface stand-alone penalty years.
            already_covered = (
                float(row.get("pension_lump_sum", 0) or 0) > 0
                or float(row.get("annuity_lump_sum", 0) or 0) > 0
            )
            if already_covered:
                continue
            cal_year = int(row["year"])
            a_age = int(row["spouse_a_age"])
            rows.append(
                f"| {cal_year} | A age {a_age} | §72(t)/(q) penalty "
                f"| ${float(val):,.0f} | early-distribution surtax |"
            )

    if not rows:
        return []

    out.append("### Major lifetime income / lump-sum events")
    out.append("")
    out.append(
        "_One-time pension and annuity events in the optimized plan, "
        "with their tax treatment. §72(t) (qualified plans) and §72(q) "
        "(non-qualified annuities) impose a 10% surtax on the taxable "
        "portion of distributions before age 59½._"
    )
    out.append("")
    out.append("| Year | Spouse | Event | Amount | Tax treatment |")
    out.append("|---|---|---|---:|---|")
    out.extend(rows)
    out.append("")
    return out


def _widow_paragraph(
    cfg: Config, inputs: Inputs, w_df: pd.DataFrame
) -> list[str]:
    """Render a widow's-penalty risk summary if mortality fires inside
    the simulation horizon.

    The simulator already encodes the year-of-death MFJ rule (TC-1)
    and switches to single filing the year after the first death. We
    surface the rate / tax delta between the last MFJ year and the
    first single year so the reader sees the survivor compression
    quantified instead of buried in §7's year-by-year detail.
    """
    horizon = cfg.horizon_age - inputs.spouse_a_age_start + 1
    death_a = cfg.mortality.year_of_death_a
    death_b = cfg.mortality.year_of_death_b
    # When either spouse has no scheduled death (None == "lives past
    # horizon") fall back to a sentinel beyond the horizon so the
    # arithmetic still works.
    da = death_a if death_a is not None else horizon + 1
    db = death_b if death_b is not None else horizon + 1
    first_death = min(da, db)
    second_death = max(da, db)
    if first_death >= second_death:
        return []
    if first_death >= horizon - 1:
        return []

    statuses = w_df["filing_status"].tolist()
    widow_idx: int | None = None
    for i in range(1, len(statuses)):
        if statuses[i] != statuses[i - 1] and statuses[i] != "mfj":
            widow_idx = i
            break
    if widow_idx is None or widow_idx >= len(w_df):
        return []

    last_mfj = w_df.iloc[widow_idx - 1]
    first_widow = w_df.iloc[widow_idx]
    surviving = "Spouse B" if da < db else "Spouse A"
    deceased = "Spouse A" if surviving == "Spouse B" else "Spouse B"

    out = ["### Widow's-penalty risk", ""]
    out.append(
        f"_{deceased} dies year {first_death} "
        f"(calendar {cfg.start_year + first_death}). "
        f"Filing status switches to single in year {widow_idx} "
        f"(calendar {cfg.start_year + widow_idx}). "
        f"Survivor ({surviving}) files single for the remaining "
        f"{horizon - widow_idx} year(s)._"
    )
    out.append("")
    out.append("| Metric | Last MFJ year | First single year | Δ |")
    out.append("|---|---:|---:|---:|")
    d_marg = float(first_widow["marginal"]) - float(last_mfj["marginal"])
    d_fed = float(first_widow["federal_tax"]) - float(last_mfj["federal_tax"])
    d_agi = float(first_widow["agi"]) - float(last_mfj["agi"])
    out.append(
        f"| AGI | ${float(last_mfj['agi']):,.0f} | "
        f"${float(first_widow['agi']):,.0f} | ${d_agi:+,.0f} |"
    )
    out.append(
        f"| Peak marginal rate | {float(last_mfj['marginal']):.0%} | "
        f"{float(first_widow['marginal']):.0%} | {d_marg * 100:+.1f}pp |"
    )
    out.append(
        f"| Federal tax | ${float(last_mfj['federal_tax']):,.0f} | "
        f"${float(first_widow['federal_tax']):,.0f} | ${d_fed:+,.0f} |"
    )
    out.append("")
    return out


def _sunset_paragraph(
    cfg: Config, inputs: Inputs, w_df: pd.DataFrame
) -> list[str]:
    """Quantify the tax-regime change configured on `cfg`.

    Pulls the last-year-of-old-regime and first-year-of-new-regime rows
    from the winning plan so the reader sees the bracket jump in
    dollars and percentage points, not just regime names.
    """
    if cfg.regime_change_year_offset is None or cfg.regime_change_target is None:
        return []
    offset = cfg.regime_change_year_offset
    if offset <= 0 or offset >= len(w_df):
        return []
    pre = w_df.iloc[offset - 1]
    post = w_df.iloc[offset]
    out = ["### Tax-regime change", ""]
    out.append(
        f"_Federal regime steps from `{cfg.tax_regime.name}` to "
        f"`{cfg.regime_change_target.name}` in year {offset} "
        f"(calendar {cfg.start_year + offset})._"
    )
    out.append("")
    out.append("| Metric | Last year of old regime | First year of new regime | Δ |")
    out.append("|---|---:|---:|---:|")
    out.append(
        f"| Calendar year | {int(pre['year'])} | {int(post['year'])} | — |"
    )
    out.append(
        f"| AGI | ${float(pre['agi']):,.0f} | ${float(post['agi']):,.0f} "
        f"| ${float(post['agi']) - float(pre['agi']):+,.0f} |"
    )
    out.append(
        f"| Peak marginal rate | {float(pre['marginal']):.0%} | "
        f"{float(post['marginal']):.0%} | "
        f"{(float(post['marginal']) - float(pre['marginal'])) * 100:+.1f}pp |"
    )
    out.append(
        f"| Federal tax | ${float(pre['federal_tax']):,.0f} | "
        f"${float(post['federal_tax']):,.0f} | "
        f"${float(post['federal_tax']) - float(pre['federal_tax']):+,.0f} |"
    )
    out.append("")
    return out


def _tldr_section(
    winner: str,
    results: dict[str, StrategyResult],
    base_cfg: Config,
    base_inputs: Inputs,
    mc: MonteCarloResult | None,
) -> list[str]:
    """Render the headline verdict, lever-change summary, and key risk
    metrics. The intent is that a reader who only reads the first 15
    lines of the report still leaves with the right takeaways."""
    w = results[winner]
    w_sum = w.summary
    base_sum = results["S0_baseline"].summary if "S0_baseline" in results else {}
    out = ["## TL;DR", ""]

    if base_sum:
        delta = w_sum["terminal_after_tax"] - base_sum["terminal_after_tax"]
        pct = (
            (delta / base_sum["terminal_after_tax"] * 100)
            if base_sum["terminal_after_tax"]
            else 0.0
        )
        if delta > 0:
            verdict = (
                f"**Verdict:** `{winner}` beats `S0_baseline` by "
                f"**${delta:,.0f}** ({pct:+.1f}%) in terminal after-tax NW."
            )
        elif delta < 0:
            verdict = (
                f"**Verdict:** `{winner}` lags `S0_baseline` by "
                f"${abs(delta):,.0f} ({pct:+.1f}%) — your current setup "
                f"already looks optimal."
            )
        else:
            verdict = (
                f"**Verdict:** `{winner}` matches `S0_baseline`; no "
                f"actionable change found by the optimizer."
            )
        out.append(verdict)
    else:
        out.append(
            f"**Winning strategy:** `{winner}` — terminal after-tax NW "
            f"${w_sum['terminal_after_tax']:,.0f}."
        )
    out.append("")

    changes = _lever_changes(base_cfg, base_inputs, w.cfg, w.inputs)
    if changes:
        out.append("**Levers the optimizer wants you to change:**")
        for ch in changes:
            if ch["kind"] == "pct":
                out.append(
                    f"- {ch['label']}: {ch['before']:.0%} → **{ch['after']:.0%}**"
                )
            else:
                out.append(
                    f"- {ch['label']}: {ch['before']} → **{ch['after']}**"
                )
    else:
        out.append(
            "_No lever changes recommended: your current inputs already "
            "match the optimizer's choice on every decision axis._"
        )
    out.append("")

    out.append("**Key risk readings:**")
    if mc is not None:
        s = mc.summary()
        tag = "safe" if s["prob_success"] >= 0.9 else "**watch — below 90%**"
        out.append(
            f"- P(success) under {s['n_paths']} stochastic paths: "
            f"**{s['prob_success']:.1%}** ({tag})"
        )
        out.append(f"- CVaR(10%) terminal NW: ${s['cvar_terminal_p10']:,.0f}")
    out.append(f"- Peak federal marginal rate: **{w_sum['peak_marginal']:.0%}**")
    tax_line = f"- Lifetime federal tax NPV: ${w_sum['lifetime_tax_npv']:,.0f}"
    if base_sum:
        d = w_sum["lifetime_tax_npv"] - base_sum["lifetime_tax_npv"]
        if abs(d) >= 1:
            tax_line += f" ({d:+,.0f} vs S0)"
    out.append(tax_line)
    irmaa_line = f"- Lifetime IRMAA NPV: ${w_sum['lifetime_irmaa_npv']:,.0f}"
    if base_sum:
        d = w_sum["lifetime_irmaa_npv"] - base_sum["lifetime_irmaa_npv"]
        if abs(d) >= 1:
            irmaa_line += f" ({d:+,.0f} vs S0)"
    out.append(irmaa_line)
    out.append("")
    return out


def build_action_report(
    cfg: Config,
    inputs: Inputs,
    results: dict[str, StrategyResult],
    sens_df: pd.DataFrame,
    base_terminal: float,
    mc: MonteCarloResult | None = None,
    *,
    scenario_path: str | None = None,
    extra_mc: dict[str, MonteCarloResult] | None = None,
    year_table_scope: str = "full",
) -> str:
    """Render the deterministic + (optional) Monte Carlo action plan as markdown.

    Parameters mirror the notebook's `build_action_report`. Returns a
    single markdown string (no trailing newline).

    Parameters
    ----------
    year_table_scope:
        Controls how many rows show up in §7 ("Year-by-year withdrawal
        & conversion plan"). One of:

        * ``"full"`` (default) — show every simulated year from start
          age through horizon. Pre-retirement years carry meaningful
          AGI / federal-tax / state-tax / any-pre-retirement-conversion
          columns even though withdrawal columns are mostly empty.
        * ``"retirement"`` — show only retirement years (the historical
          v1–v6 behavior). Useful when the report is being shared as a
          condensed deliverable and the working-year tax detail is not
          interesting to the audience.

        Unknown values raise ``ValueError``.
    """
    if year_table_scope not in {"full", "retirement"}:
        raise ValueError(
            f"year_table_scope must be 'full' or 'retirement', "
            f"got {year_table_scope!r}."
        )
    winner = _winning_strategy(results)
    w = results[winner]
    w_cfg, w_inputs, w_df, w_sum = w.cfg, w.inputs, w.df, w.summary
    horizon = cfg.horizon_age - inputs.spouse_a_age_start + 1
    starting_total = inputs.starting.total_excl_real_estate
    household_wages = (
        inputs.income.spouse_a_gross
        + inputs.income.spouse_b_gross
        + inputs.income.spouse_a_bonus
    )
    peak = _peak_marginal_year(w_df)
    conv_start_age, conv_end_age = _conversion_window(inputs, cfg)

    md: list[str] = []
    md.append("# Retirement Tax Optimization — Action Plan")
    md.append("")
    md.append(
        f"_{horizon}-year horizon, ages {inputs.spouse_a_age_start}"
        f"/{inputs.spouse_b_age_start} to {cfg.horizon_age}, "
        f"tax regime `{cfg.tax_regime.name}`._"
    )
    if scenario_path:
        md.append("")
        md.append(f"_Scenario file: `{scenario_path}`._")
    md.append("")

    md.extend(_tldr_section(winner, results, cfg, inputs, mc))

    # ---- 1. Household snapshot --------------------------------------------
    md.append("## 1. Household snapshot")
    md.append("")
    md.append("| Item | Value |")
    md.append("|---|---:|")
    md.append(
        f"| Spouse A age (retire / SS) | {inputs.spouse_a_age_start} "
        f"({inputs.spouse_a_retire_age} / {inputs.ss.effective_start_age_a}) |"
    )
    md.append(
        f"| Spouse B age (retire / SS) | {inputs.spouse_b_age_start} "
        f"({inputs.spouse_b_retire_age} / {inputs.ss.effective_start_age_b}) |"
    )
    md.append(f"| Combined gross W-2 income | ${household_wages:,.0f} |")
    # Use the resolved spending profile's base, not `inputs.annual_expenses`.
    # The simulator drives spending off `cfg.resolved_spending()`, so a
    # scenario that sets `cfg.spending.base_spending` but leaves
    # `inputs.annual_expenses` at the default would otherwise mislabel
    # this row.
    resolved_spending = cfg.resolved_spending()
    md.append(
        f"| Annual expenses (today's $) | ${resolved_spending.base_spending:,.0f} |"
    )
    md.append(f"| Total liquid + retirement assets | ${starting_total:,.0f} |")
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Spouse A pretax (401k + IRA) | "
        f"${inputs.starting.spouse_a_pretax_401k + inputs.starting.spouse_a_pretax_ira:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Spouse B pretax (401k + IRA) | "
        f"${inputs.starting.spouse_b_pretax_401k + inputs.starting.spouse_b_pretax_ira:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Roth (both spouses, pooled) | "
        f"${inputs.starting.spouse_a_roth_ira + inputs.starting.spouse_b_roth_ira:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; Taxable brokerage | "
        f"${inputs.starting.taxable_brokerage:,.0f} |"
    )
    md.append(
        f"| &nbsp;&nbsp;&nbsp; HSA / pension cash-balance | "
        f"${inputs.starting.hsa:,.0f} / ${inputs.starting.pension_balance:,.0f} |"
    )
    if (
        inputs.spouse_a_employer_match_rate > 0
        or inputs.spouse_b_employer_match_rate > 0
    ):
        a_match = (
            f"{inputs.spouse_a_employer_match_rate:.0%} on first "
            f"{inputs.spouse_a_employer_match_max_pct:.0%}"
            if inputs.spouse_a_employer_match_rate > 0
            else "—"
        )
        b_match = (
            f"{inputs.spouse_b_employer_match_rate:.0%} on first "
            f"{inputs.spouse_b_employer_match_max_pct:.0%}"
            if inputs.spouse_b_employer_match_rate > 0
            else "—"
        )
        md.append(f"| Employer 401(k) match (A / B) | {a_match} / {b_match} |")
    md.append("")

    md.extend(_assumptions_block(cfg, inputs))
    md.extend(_sunset_paragraph(w_cfg, w_inputs, w_df))
    md.extend(_widow_paragraph(w_cfg, w_inputs, w_df))

    # ---- 2. Recommended plan ----------------------------------------------
    md.append("## 2. Recommended plan")
    md.append("")
    md.append(f"**Winning strategy:** `{winner}`")
    md.append("")
    md.append("| Lever | Recommended | Currently |")
    md.append("|---|---:|---:|")
    # The optimizer's three decision variables are the two Roth-401(k)
    # splits and the conversion target bracket. The total deferral
    # percentages and withdrawal strategy aren't searched over today, so
    # we only show them when they actually differ from the baseline
    # (e.g. a hand-tuned scenario or a future expanded decision space).
    if w_inputs.spouse_a_total_contrib_pct != inputs.spouse_a_total_contrib_pct:
        md.append(
            f"| Spouse A 401(k) deferral | {w_inputs.spouse_a_total_contrib_pct:.0%} of salary "
            f"| {inputs.spouse_a_total_contrib_pct:.0%} |"
        )
    md.append(
        f"| Spouse A Roth share of deferral | {w_inputs.spouse_a_roth_401k_pct:.0%} "
        f"| {inputs.spouse_a_roth_401k_pct:.0%} |"
    )
    if w_inputs.spouse_b_total_contrib_pct != inputs.spouse_b_total_contrib_pct:
        md.append(
            f"| Spouse B 401(k) deferral | {w_inputs.spouse_b_total_contrib_pct:.0%} of salary "
            f"| {inputs.spouse_b_total_contrib_pct:.0%} |"
        )
    md.append(
        f"| Spouse B Roth share of deferral | {w_inputs.spouse_b_roth_401k_pct:.0%} "
        f"| {inputs.spouse_b_roth_401k_pct:.0%} |"
    )
    md.append(
        f"| Roth conversion target bracket (gap years) "
        f"| {w_cfg.roth_conversion_target_bracket:.0%} "
        f"| {cfg.roth_conversion_target_bracket:.0%} |"
    )
    if w_cfg.withdrawal_strategy != cfg.withdrawal_strategy:
        md.append(
            f"| Withdrawal strategy in retirement | `{w_cfg.withdrawal_strategy}` "
            f"| `{cfg.withdrawal_strategy}` |"
        )
    md.append("")

    md.extend(_optimizer_rationale(cfg, inputs, w_cfg, w_inputs, w_df, w_sum))

    # ---- 3. Expected outcomes ---------------------------------------------
    md.append("## 3. Expected outcomes (deterministic, point-estimate)")
    md.append("")
    strategy_order = ["S0_baseline", "S1_all_roth", "S2_bracket_fill_22", "S3_optimizer"]
    cols = [s for s in strategy_order if s in results]
    extra = [s for s in results if s not in strategy_order]
    cols.extend(extra)

    if len(cols) > 1:
        md.append(
            "_Best value in each row is **bolded**. Higher is better for "
            "terminal NW; lower is better for tax, IRMAA, peak marginal._"
        )
        md.append("")
        header_cells = [c.replace("_", " ") for c in cols]
        md.append("| Metric | " + " | ".join(header_cells) + " |")
        md.append("|---|" + "---:|" * len(cols))

        metric_specs = [
            ("Terminal after-tax NW", "terminal_after_tax", "max", "${v:,.0f}"),
            ("Lifetime federal tax (NPV)", "lifetime_tax_npv", "min", "${v:,.0f}"),
            ("Lifetime IRMAA (NPV)", "lifetime_irmaa_npv", "min", "${v:,.0f}"),
            ("Peak federal marginal rate", "peak_marginal", "min", "{v:.0%}"),
            ("Years with IRMAA", "years_irmaa", "min", "{v:.0f}"),
            ("Peak IRMAA tier", "peak_irmaa_tier", "min", "{v:.0f}"),
        ]
        for label, key, mode, fmt in metric_specs:
            vals = [results[c].summary[key] for c in cols]
            best_idx = (
                max(range(len(vals)), key=lambda i: vals[i])
                if mode == "max"
                else min(range(len(vals)), key=lambda i: vals[i])
            )
            tied = [i for i, v in enumerate(vals) if v == vals[best_idx]]
            cells = []
            for i, v in enumerate(vals):
                text = fmt.format(v=v)
                if i in tied and len(tied) < len(vals):
                    text = f"**{text}**"
                cells.append(text)
            md.append(f"| {label} | " + " | ".join(cells) + " |")
    else:
        md.append(
            f"| Terminal after-tax NW | ${w_sum['terminal_after_tax']:,.0f} |"
        )
    md.append("")
    md.append(
        f"_Peak federal marginal year (for `{winner}`): age **{peak['age']}**, "
        f"AGI ~${peak['agi']:,.0f}, marginal **{peak['marginal']:.0%}**._"
    )
    md.append("")

    md.extend(_heir_rate_sensitivity(w_cfg, w_df))

    # ---- 4. Risk picture (Monte Carlo) ------------------------------------
    if mc is not None:
        s = mc.summary()
        md.append("## 4. Risk picture (Monte Carlo)")
        md.append("")
        md.append(
            f'_Based on {s["n_paths"]} stochastic paths under the '
            f"`{type(mc.cfg.resolved_market()).__name__}` market model._"
        )
        md.append("")
        md.append("| Metric | Value | Reading |")
        md.append("|---|---:|---|")
        md.append(
            f'| Probability of success | {s["prob_success"]:.1%} | '
            f'{"safe" if s["prob_success"] >= 0.9 else "watch — below 90%"} |'
        )
        md.append(
            f"| Terminal NW p5 / p50 / p95 | "
            f'${s["terminal_p5"]:,.0f} / ${s["terminal_p50"]:,.0f} '
            f'/ ${s["terminal_p95"]:,.0f} | spread of bad-to-good outcomes |'
        )
        md.append(
            f'| CVaR(10%) terminal | ${s["cvar_terminal_p10"]:,.0f} | '
            f"expected NW in the worst 10% of paths |"
        )
        md.append(
            f'| Median lifetime tax (NPV) | ${s["lifetime_tax_p50"]:,.0f} | — |'
        )
        if s["median_ruin_year_offset"] >= 0:
            md.append(
                f'| Median ruin year | {s["median_ruin_year_offset"]:.0f} | '
                f"**at least half of failure paths run out by year "
                f'{s["median_ruin_year_offset"]:.0f}** |'
            )
        else:
            md.append("| Median ruin year | — | no path ran out of money |")
        md.append("")

        if extra_mc:
            md.extend(_cross_model_table(mc, extra_mc))

    # ---- 5. Top sensitivity levers ----------------------------------------
    md.append("## 5. Highest-leverage levers (top 3 by tornado swing)")
    md.append("")
    md.append("| Knob | Range tested | Best direction | $ swing |")
    md.append("|---|---|---|---:|")
    for _, row in sens_df.head(3).iterrows():
        param = row["param"]
        lo, hi = row["low_value"], row["high_value"]
        d_lo, d_hi = row["delta_low"], row["delta_high"]
        # Honest direction labelling: if neither side of the tested range
        # improves on the baseline, say so. Otherwise pick the direction
        # that yields a positive delta. (Previously this always said
        # "higher" when delta_high == delta_low, producing misleading
        # "+$0" recommendations for knobs already at their boundary.)
        if d_hi <= 0 and d_lo <= 0:
            direction_cell = "at boundary in tested range (—)"
        elif d_hi >= d_lo:
            direction_cell = f"higher (+${d_hi:,.0f})"
        else:
            direction_cell = f"lower (+${d_lo:,.0f})"
        if param.endswith("_pct") or "rate" in param or param == "inflation":
            rng = f"{lo:.0%} → {hi:.0%}"
        elif param.endswith("_age"):
            rng = f"{int(lo)} → {int(hi)}"
        else:
            rng = f"${lo:,.0f} → ${hi:,.0f}"
        md.append(
            f"| `{param}` | {rng} | {direction_cell} "
            f'| ${row["swing"]:,.0f} |'
        )
    md.append("")

    # ---- 6. Year-by-year action timeline ----------------------------------
    md.append("## 6. Year-by-year action timeline")
    md.append("")
    md.extend(_this_year_actions(w_cfg, w_inputs, w_df))
    md.extend(_lifetime_events(w_cfg, w_inputs, w_df))
    md.append("### Multi-decade phases")
    md.append("")
    md.append("| Phase | Ages (Spouse A) | What to do |")
    md.append("|---|---|---|")
    md.append(
        f"| **Accumulation** | {inputs.spouse_a_age_start}–{inputs.spouse_a_retire_age - 1} | "
        f"Defer {w_inputs.spouse_a_total_contrib_pct:.0%} of A's salary "
        f"({w_inputs.spouse_a_roth_401k_pct:.0%} Roth / "
        f"{1 - w_inputs.spouse_a_roth_401k_pct:.0%} Traditional) and "
        f"{w_inputs.spouse_b_total_contrib_pct:.0%} of B's "
        f"({w_inputs.spouse_b_roth_401k_pct:.0%} Roth / "
        f"{1 - w_inputs.spouse_b_roth_401k_pct:.0%} Traditional). "
        f"Max HSA. Build 1–2 years' expenses in taxable as IRMAA buffer. |"
    )
    if conv_start_age <= conv_end_age and w_cfg.roth_conversion_target_bracket > 0:
        md.append(
            f"| **Gap years (conversion window)** | {conv_start_age}–{conv_end_age} | "
            f"No wages and SS not yet started → fill the "
            f"{w_cfg.roth_conversion_target_bracket:.0%} bracket each year via "
            f"pretax → Roth conversions. Pay conversion tax from taxable. |"
        )
    else:
        md.append(
            f"| **Gap years** | {conv_start_age}–{conv_end_age} | "
            f"Optimizer chose 0% conversion target — keep gap-year income low and "
            f"rely on taxable + Roth withdrawals. |"
        )
    md.append(
        "| **Medicare / IRMAA** | 65+ | "
        "IRMAA tiers depend on AGI from **2 years prior**. "
        "Watch the 22% / 24% bracket and the IRMAA cliffs at 65, 67. |"
    )
    md.append(
        f"| **SS claim** | {inputs.ss.start_age} | "
        f"Both spouses begin Social Security "
        f"(${(inputs.ss.monthly_spouse_a + inputs.ss.monthly_spouse_b) * 12:,.0f}/yr today). "
        f"After this, conversion-window income includes SS. |"
    )
    md.append(
        f"| **RMDs begin** | {cfg.rmd_start_age} | "
        f"Required minimum distributions hit Spouse A's pretax balance. "
        f"Spouse B's RMDs follow on B's schedule. Reduce or stop conversions. |"
    )
    md.append(
        f"| **Drawdown** | {cfg.rmd_start_age + 1}–{cfg.horizon_age} | "
        f"Withdrawal sequence: `{w_cfg.withdrawal_strategy}`. "
        f"Take RMDs first, then top up from taxable / Roth as needed. |"
    )
    md.append("")

    # ---- 7. Year-by-year withdrawal & conversion plan ---------------------
    md.append("## 7. Year-by-year withdrawal & conversion plan")
    md.append("")
    retire_age = min(inputs.spouse_a_retire_age, inputs.spouse_b_retire_age)
    if year_table_scope == "retirement":
        md.append(
            "_Retirement years only. Pre-retirement years are pure "
            "accumulation (contributions, no withdrawals or conversions). "
            "Cells with no activity render as `—`. Pass "
            "`year_table_scope=\"full\"` to `build_action_report` (or "
            "`--year-table-scope full` to the CLI) to see the entire "
            "horizon, including working years._"
        )
    else:
        md.append(
            "_Full horizon: every simulated year from start age through "
            "the horizon. Pre-retirement rows (below the row marked "
            f"**RETIRE @ {retire_age}**) carry meaningful AGI / federal "
            "tax / state tax columns even though withdrawal columns are "
            "mostly empty — useful to see when contributions phase into "
            "drawdown and to spot any pre-retirement Roth conversions. "
            "Cells with no activity render as `—`. Pass "
            "`year_table_scope=\"retirement\"` (or "
            "`--year-table-scope retirement`) to collapse to "
            "retirement-only._"
        )
    md.append("")

    def _money(v: float | None) -> str:
        # Use `round(...) == 0` instead of `== 0.0` so floats that are
        # numerically near zero (e.g. 1e-9 from float arithmetic in the
        # cascade) also render as `—`. Without this, the table shows
        # confusing `$0` cells next to `—` cells for what are logically
        # the same "nothing happened" outcome.
        if v is None:
            return "—"
        f = float(v)
        if round(f) == 0:
            return "—"
        return f"${f:,.0f}"

    if year_table_scope == "retirement":
        retire_rows = w_df[w_df["spouse_a_age"] >= retire_age]
    else:
        retire_rows = w_df

    # Conditional columns: only include if relevant for this scenario.
    # State tax is rendered only when a non-stateless regime is active
    # (otherwise the column is uniformly $0 / —). Healthcare collapses
    # Medicare-base + pre-65 health − ACA-credit into a single $ column
    # (separate from IRMAA, which stays in its own column since it's a
    # discrete cliff lever the user manages directly).
    has_state = (
        cfg.state_regime.name.lower() != "stateless"
        and retire_rows.get("state_tax", pd.Series([0])).abs().sum() > 0
    )

    def _health_of(row: pd.Series) -> float:
        base = float(row.get("medicare_base_premium", 0) or 0)
        pre65 = float(row.get("health_pre65", 0) or 0)
        aca = float(row.get("aca_apt_credit", 0) or 0)
        return max(0.0, base + pre65 - aca)

    has_health = any(_health_of(r) > 0 for _, r in retire_rows.iterrows())

    headers = ["Age (A)", "Pretax W/D", "Roth conv.", "Roth W/D",
               "Taxable W/D", "RMD", "AGI", "Fed tax"]
    if has_state:
        headers.append("State tax")
    headers.extend(["IRMAA"])
    if has_health:
        headers.append("Health $")

    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "---:|" * len(headers))

    retirement_marker_emitted = False
    for _, r in retire_rows.iterrows():
        age = int(r["spouse_a_age"])
        # When showing the full horizon, drop a visual marker row right
        # before the first retirement year so the reader can see at a
        # glance where accumulation ends and drawdown begins.
        if (
            year_table_scope == "full"
            and not retirement_marker_emitted
            and age >= retire_age
        ):
            marker = (
                f"| **RETIRE @ {retire_age}** "
                + "| " * (len(headers) - 1)
                + "|"
            )
            md.append(marker)
            retirement_marker_emitted = True
        row_cells = [
            f"{age}",
            _money(r["pretax_withdrawal"]),
            _money(r["roth_conversion"]),
            _money(r["roth_withdrawal"]),
            _money(r["taxable_withdrawal"]),
            _money(r["rmd"]),
            _money(r["agi"]),
            _money(r["federal_tax"]),
        ]
        if has_state:
            row_cells.append(_money(r.get("state_tax", 0)))
        row_cells.append(_money(r["irmaa"]))
        if has_health:
            row_cells.append(_money(_health_of(r)))
        md.append("| " + " | ".join(row_cells) + " |")
    md.append("")
    legend = (
        "_Pretax W/D = combined Spouse A + B pretax withdrawals; RMD = "
        "the IRS-required floor (A and B combined). Roth conversions show "
        "up as AGI in the same row but produce no cash to the household — "
        "they shift dollars from pretax → Roth and the conversion tax is "
        "paid from taxable."
    )
    if has_health:
        legend += (
            " **Health $** = Medicare base premium + pre-65 healthcare − "
            "ACA premium tax credit; IRMAA stays as a separate column "
            "because it's a discrete AGI-cliff lever you can manage."
        )
    legend += "_"
    md.append(legend)
    md.append("")

    # ---- 8. Always-good hygiene -------------------------------------------
    md.append("## 8. Always-good hygiene")
    md.append("")
    md.append(
        "- **Max the HSA each year** (triple tax-advantaged: deductible, tax-free "
        "growth, tax-free qualified withdrawals)."
    )
    md.append(
        "- **Hold 1–2 years of expenses in taxable** as an IRMAA / sequence-of-returns "
        "buffer; it lets you avoid IRA withdrawals in down-market years and to "
        "manage AGI around IRMAA cliffs."
    )
    md.append(
        "- **Re-run this annually** with updated balances, salaries, and "
        "tax-regime assumptions. Roth-conversion targets in particular are very "
        "sensitive to bracket changes."
    )
    md.append(
        "- **Beneficiary review:** primary + contingent on every account, including "
        "the HSA (which becomes ordinary income to a non-spouse beneficiary)."
    )
    md.append("")

    # ---- 9. Caveats -------------------------------------------------------
    md.append("## 9. Caveats — what this plan does NOT model")
    md.append("")
    md.append(
        "- **State income tax** outside the bundled `STATELESS / CA / NY / IL / MA` "
        "presets. Add a custom `StateTaxRegime` for your state if it's not in this list."
    )
    md.append(
        "- **Estate / inherited-IRA dynamics** beyond the flat `heir_marginal_rate` "
        "haircut on terminal pretax + HSA balances. SECURE Act 10-year drawdown "
        "timing for non-spouse heirs is not modeled."
    )
    md.append("- **Health-care shocks** beyond a single optional LTC stress test.")
    md.append(
        "- **Tax-law revisions** beyond the bundled regimes (`TCJA_EXTENDED`, "
        "`PRE_TCJA_2017`, `SUNSET_2026`). Re-run with `cfg.tax_regime = SUNSET_2026` "
        "to stress-test current-law sunset."
    )
    md.append("")
    md.append(
        "_This is decision-support output, not tax/legal/investment advice. "
        "Consult a qualified professional before acting._"
    )

    return "\n".join(md)


# ---------------------------------------------------------------------
# Multi-scenario diff API (R-C.3)
# ---------------------------------------------------------------------


def compare_scenarios(
    scenarios: list[tuple[str, Config, Inputs]],
    *,
    mc: dict[str, MonteCarloResult] | None = None,
) -> str:
    """Render a side-by-side markdown diff of N independent scenarios.

    Useful for "what if I move to CA vs NY", "what if I retire at 62
    vs 67", or "what if TCJA sunsets vs stays extended" — anywhere you
    want to see two or more household configurations against the same
    metric set without scrolling between two full action reports.

    Each scenario is identified by a short name (rendered as the column
    header) and produces one column. Best value in each row is bolded.

    Parameters
    ----------
    scenarios:
        A list of `(name, cfg, inputs)` triples. Order is preserved in
        the rendered table. Names must be unique.
    mc:
        Optional `{name: MonteCarloResult}` dict; when provided, the
        comparison adds a Monte Carlo sub-section with P(success) and
        CVaR(10%) per scenario.

    Returns
    -------
    A self-contained markdown string suitable for terminal / GitHub /
    email rendering.
    """
    if not scenarios:
        return "_compare_scenarios: no scenarios provided._"
    names = [n for n, _c, _i in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("compare_scenarios: scenario names must be unique")

    summaries: list[dict] = []
    cfgs: list[Config] = []
    for _name, cfg, inputs in scenarios:
        df = simulate(cfg, inputs)
        summaries.append(summarize(df, heir_marginal_rate=cfg.heir_marginal_rate))
        cfgs.append(cfg)

    md: list[str] = []
    md.append("# Scenario comparison")
    md.append("")
    md.append(
        f"_{len(scenarios)} household scenarios simulated independently. "
        "Best value in each row is **bolded**. Higher is better for "
        "terminal NW; lower is better for tax, IRMAA, peak marginal._"
    )
    md.append("")
    md.append("## Outcome metrics")
    md.append("")
    md.append("| Metric | " + " | ".join(names) + " |")
    md.append("|---|" + "---:|" * len(names))

    metric_specs = [
        ("Terminal after-tax NW", "terminal_after_tax", "max", "${v:,.0f}"),
        ("Lifetime federal tax (NPV)", "lifetime_tax_npv", "min", "${v:,.0f}"),
        ("Lifetime IRMAA (NPV)", "lifetime_irmaa_npv", "min", "${v:,.0f}"),
        ("Peak federal marginal rate", "peak_marginal", "min", "{v:.0%}"),
        ("Years with IRMAA", "years_irmaa", "min", "{v:.0f}"),
    ]
    for label, key, mode, fmt in metric_specs:
        vals = [s[key] for s in summaries]
        best = (
            max(range(len(vals)), key=lambda i: vals[i])
            if mode == "max"
            else min(range(len(vals)), key=lambda i: vals[i])
        )
        tied = [i for i, v in enumerate(vals) if v == vals[best]]
        cells = []
        for i, v in enumerate(vals):
            text = fmt.format(v=v)
            if i in tied and len(tied) < len(vals):
                text = f"**{text}**"
            cells.append(text)
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Scenario assumptions")
    md.append("")
    md.append("| Assumption | " + " | ".join(names) + " |")
    md.append("|---|" + "---|" * len(names))
    rows: list[tuple[str, list[str]]] = []
    rows.append(
        ("Heir marginal rate", [f"{c.heir_marginal_rate:.0%}" for c in cfgs])
    )
    rows.append(
        ("Federal regime", [f"`{c.tax_regime.name}`" for c in cfgs])
    )

    def _change(c: Config) -> str:
        if c.regime_change_year_offset is None:
            return "—"
        return f"→ `{c.regime_change_target.name}` yr {c.regime_change_year_offset}"

    rows.append(("Regime change", [_change(c) for c in cfgs]))
    rows.append(("State regime", [f"`{c.state_regime.name}`" for c in cfgs]))

    def _mortality(c: Config) -> str:
        ma = c.mortality.year_of_death_a
        mb = c.mortality.year_of_death_b
        return f"A:{ma if ma is not None else '∞'} / B:{mb if mb is not None else '∞'}"

    rows.append(("Mortality (yr_a / yr_b)", [_mortality(c) for c in cfgs]))
    rows.append(
        ("Step-up at first death", ["on" if c.stepup_at_first_death else "off" for c in cfgs])
    )
    rows.append(
        ("ACA enabled", ["on" if c.aca_enabled else "off" for c in cfgs])
    )
    rows.append(
        ("Market model", [type(c.market).__name__ for c in cfgs])
    )
    rows.append(
        ("Inflation", [f"{c.inflation:.1%}" for c in cfgs])
    )
    for label, cells in rows:
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("")

    if mc:
        md.append("## Risk picture (Monte Carlo)")
        md.append("")
        mc_names = [n for n in names if n in mc]
        if not mc_names:
            md.append("_No Monte Carlo results provided for any scenario._")
            md.append("")
        else:
            md.append(
                f"_From the `{', '.join(mc_names)}` MC run(s)._"
            )
            md.append("")
            md.append("| Metric | " + " | ".join(mc_names) + " |")
            md.append("|---|" + "---:|" * len(mc_names))
            ps_vals = [mc[n].prob_success() for n in mc_names]
            best_ps = max(range(len(ps_vals)), key=lambda i: ps_vals[i])
            ps_cells = [
                (f"**{v:.1%}**" if i == best_ps and len(set(ps_vals)) > 1 else f"{v:.1%}")
                for i, v in enumerate(ps_vals)
            ]
            md.append("| P(success) | " + " | ".join(ps_cells) + " |")
            cvar_vals = [mc[n].cvar_terminal(0.10) for n in mc_names]
            best_cvar = max(range(len(cvar_vals)), key=lambda i: cvar_vals[i])
            cvar_cells = [
                (f"**${v:,.0f}**" if i == best_cvar and len(set(cvar_vals)) > 1 else f"${v:,.0f}")
                for i, v in enumerate(cvar_vals)
            ]
            md.append("| CVaR(10%) terminal | " + " | ".join(cvar_cells) + " |")
            md.append("")

    md.append(
        "_Each scenario is independent — there is no shared optimization. "
        "To find the best plan inside a scenario, run "
        "`optimize_household(...)` first and pass the optimized cfg/inputs in._"
    )
    return "\n".join(md)
