"""Declarative schema for every editable scenario field.

`FIELD_SCHEMA` is the single source of truth used to:

  * build the Simple and Advanced form tabs,
  * map form values back into a scenario JSON dict,
  * map a loaded scenario dict back into form values.

Each `FormField` declares:

  * `path`     - dotted path inside the scenario JSON, e.g.
                 "config.market.equity_mu" or
                 "inputs.starting.spouse_a_pretax_401k".
  * `label`    - human-readable form label.
  * `kind`     - one of "number", "percent", "int", "bool", "select", "text".
                 "percent" is just a number rendered with a "%" suffix; the
                 stored / loaded value is always a float (0.07, not 7).
  * `options`  - for "select", a list of `(value, label)` tuples.
  * `group`    - section heading inside the form.
  * `tier`     - "simple" or "advanced".
  * `step`     - optional step for `dcc.Input(type="number")`.
  * `min` / `max` - optional bounds (display-only; not enforced server-side).
  * `help`     - optional one-line description shown as the input's tooltip.

Markets and spending profiles are *discriminated unions* keyed on `kind`.
The schema lists every block's fields; `state.py` shapes them into the
nested JSON the simulator expects based on the active discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FormField:
    path: str
    label: str
    kind: str = "number"
    options: tuple[tuple[Any, str], ...] = ()
    group: str = "Misc"
    tier: str = "advanced"
    step: float | None = None
    min: float | None = None
    max: float | None = None
    help: str | None = None
    # ``couple_only`` flags fields that only apply to a married
    # household. When ``inputs.household_kind == "single"`` the Dash
    # callback ``_toggle_couple_only_inputs`` disables every input
    # whose schema entry has ``couple_only=True`` (and re-enables them
    # on toggle back to "mfj"). The simulator already silently ignores
    # spouse-B inputs when single, so this is purely a UX affordance —
    # the values stay in the scenario JSON so a flip back to "mfj"
    # restores them without re-entry.
    couple_only: bool = False

    @property
    def component_id(self) -> dict[str, str]:
        """ID used for Dash pattern-matching callbacks.

        We use a `{type, path}` dict so a single callback can read every
        form input via `Input({"type": "form-input", "path": ALL}, "value")`.
        """
        return {"type": "form-input", "path": self.path}


# --- Choice tables ---------------------------------------------------

_HOUSEHOLD_KINDS: tuple[tuple[str, str], ...] = (
    ("mfj", "Married filing jointly (two spouses)"),
    ("single", "Single (one filer)"),
)

_TAX_REGIMES: tuple[tuple[str, str], ...] = (
    ("tcja", "TCJA (current law, 2018-2025)"),
    ("sunset", "Sunset / 2026 brackets"),
    ("pre_tcja", "Pre-TCJA (2017 brackets)"),
)

_STATE_REGIMES: tuple[tuple[str, str], ...] = (
    ("stateless", "Stateless (FL, TX, WA, NV, ...)"),
    ("ca", "California"),
    ("ny", "New York"),
    ("il", "Illinois"),
    ("ma", "Massachusetts"),
)

_REGIMES_NULLABLE: tuple[tuple[str | None, str], ...] = (
    (None, "(no regime swap)"),
    ("tcja", "TCJA"),
    ("sunset", "Sunset"),
    ("pre_tcja", "Pre-TCJA"),
)

_STATE_REGIMES_NULLABLE: tuple[tuple[str | None, str], ...] = (
    (None, "(no state regime swap)"),
    ("stateless", "Stateless"),
    ("ca", "California"),
    ("ny", "New York"),
    ("il", "Illinois"),
    ("ma", "Massachusetts"),
)

_WITHDRAWAL_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("conventional", "Conventional (taxable -> pretax -> Roth)"),
    ("proportional", "Proportional"),
    ("bracket_fill", "Bracket-fill"),
)

_MARKET_KINDS: tuple[tuple[str, str], ...] = (
    ("lognormal", "Lognormal (stochastic)"),
    ("deterministic", "Deterministic (constant returns)"),
    ("bootstrap", "Bootstrap (block-resample empirical)"),
    ("historical_sequence", "Historical sequence"),
)

_SPENDING_KINDS: tuple[tuple[str, str], ...] = (
    ("flat", "Flat (inflation-indexed)"),
    ("smile", "Retirement smile (slow / go-go / no-go)"),
)


# --- Schema ---------------------------------------------------------

def _f(*args: Any, **kwargs: Any) -> FormField:
    return FormField(*args, **kwargs)


FIELD_SCHEMA: tuple[FormField, ...] = (
    # --- Household kind (Simple) ---
    # First field on purpose: it discriminates which downstream tax
    # tables / FICA thresholds / IRMAA tiers the simulator uses, and
    # the Dash callback grays out every couple-only input below when
    # set to "single".
    _f("inputs.household_kind", "Filing status", "select",
       options=_HOUSEHOLD_KINDS,
       group="Ages & retirement", tier="simple",
       help="MFJ uses married-filing-jointly tax tables and reads both "
            "spouses' inputs. Single uses single tables, single FICA "
            "Additional-Medicare threshold, single IRMAA tiers, and "
            "ignores every Spouse-B field below."),
    _f("inputs.spouse_a_age_start", "Spouse A current age", "int",
       group="Ages & retirement", tier="simple", min=18, max=100,
       help="Spouse A's age in years at the start of the simulation. "
            "Drives RMD start, Medicare eligibility, and SS claim windows."),
    _f("inputs.spouse_b_age_start", "Spouse B current age", "int",
       group="Ages & retirement", tier="simple", min=18, max=100,
       couple_only=True,
       help="Spouse B's age in years at the start of the simulation. "
            "Drives RMD start, Medicare eligibility, and SS claim windows."),
    _f("inputs.spouse_a_retire_age", "Spouse A retire age", "int",
       group="Ages & retirement", tier="simple", min=40, max=100,
       help="Age at which Spouse A stops earning W-2 income. Set equal "
            "to current age to model an already-retired spouse."),
    _f("inputs.spouse_b_retire_age", "Spouse B retire age", "int",
       group="Ages & retirement", tier="simple", min=40, max=100,
       couple_only=True,
       help="Age at which Spouse B stops earning W-2 income. Set equal "
            "to current age to model an already-retired spouse."),

    # --- Household: income (Simple) ---
    _f("inputs.income.spouse_a_gross", "Spouse A gross W-2", "number",
       group="Current income", tier="simple", step=1000,
       help="Annual W-2 gross wages for Spouse A in today's dollars, "
            "before any pretax deductions. Inflated by wage growth each year."),
    _f("inputs.income.spouse_b_gross", "Spouse B gross W-2", "number",
       group="Current income", tier="simple", step=1000,
       couple_only=True,
       help="Annual W-2 gross wages for Spouse B in today's dollars, "
            "before any pretax deductions. Inflated by wage growth each year."),
    _f("inputs.income.spouse_a_bonus", "Spouse A bonus", "number",
       group="Current income", tier="simple", step=1000,
       help="Annual cash bonus for Spouse A in today's dollars. Treated "
            "as ordinary income at the marginal rate."),
    _f("inputs.income.interest", "Taxable interest", "number",
       group="Current income", tier="simple", step=100,
       help="Annual taxable interest from non-retirement accounts (savings, "
            "CDs, taxable bonds). Taxed as ordinary income."),
    _f("inputs.income.capital_gains", "Realized capital gains", "number",
       group="Current income", tier="simple", step=100,
       help="Annual realized long-term capital gains in today's dollars. "
            "Taxed at LTCG rates; ignored when withdrawal strategy generates "
            "synthetic gains."),
    _f("inputs.income.dividends", "Dividends", "number",
       group="Current income", tier="simple", step=100,
       help="Annual dividends from taxable accounts in today's dollars. The "
            "qualified-fraction is taxed at LTCG rates; the rest is ordinary."),

    # --- Household: contributions (Simple) ---
    _f("inputs.spouse_a_total_contrib_pct", "Spouse A total 401(k) %",
       "percent", group="401(k) contributions", tier="simple",
       help="Fraction of Spouse A's gross wages contributed to the 401(k). "
            "Decimal: 0.10 = 10%. Capped at the IRS elective-deferral limit."),
    _f("inputs.spouse_b_total_contrib_pct", "Spouse B total 401(k) %",
       "percent", group="401(k) contributions", tier="simple",
       couple_only=True,
       help="Fraction of Spouse B's gross wages contributed to the 401(k). "
            "Decimal: 0.10 = 10%. Capped at the IRS elective-deferral limit."),
    _f("inputs.spouse_a_roth_401k_pct", "Spouse A Roth-401(k) %",
       "percent", group="401(k) contributions", tier="simple",
       help="Fraction of Spouse A's 401(k) contribution routed to Roth-401(k). "
            "Rest goes to pretax. 0.0 = all pretax, 1.0 = all Roth."),
    _f("inputs.spouse_b_roth_401k_pct", "Spouse B Roth-401(k) %",
       "percent", group="401(k) contributions", tier="simple",
       couple_only=True,
       help="Fraction of Spouse B's 401(k) contribution routed to Roth-401(k). "
            "Rest goes to pretax. 0.0 = all pretax, 1.0 = all Roth."),
    _f("inputs.spouse_a_employer_match_rate", "Spouse A match rate",
       "percent", group="401(k) contributions", tier="simple",
       help="Employer match rate for Spouse A. 1.0 = dollar-for-dollar; "
            "0.5 = 50 cents on the dollar; 0.0 = no match."),
    _f("inputs.spouse_a_employer_match_max_pct", "Spouse A match max %",
       "percent", group="401(k) contributions", tier="simple",
       help="Employer-match cap on contribution percentage for Spouse A. "
            "0.07 = match applies only to first 7% of salary."),
    _f("inputs.spouse_b_employer_match_rate", "Spouse B match rate",
       "percent", group="401(k) contributions", tier="simple",
       couple_only=True,
       help="Employer match rate for Spouse B. 1.0 = dollar-for-dollar; "
            "0.5 = 50 cents on the dollar; 0.0 = no match."),
    _f("inputs.spouse_b_employer_match_max_pct", "Spouse B match max %",
       "percent", group="401(k) contributions", tier="simple",
       couple_only=True,
       help="Employer-match cap on contribution percentage for Spouse B. "
            "0.07 = match applies only to first 7% of salary."),
    _f("inputs.contrib.hsa_family", "HSA family contribution", "number",
       group="401(k) contributions", tier="simple", step=100,
       help="Annual HSA contribution in today's dollars. IRS family cap "
            "($8,300 for 2024) is the upper bound; over-contribution is clipped."),

    # --- Household: starting balances (Simple) ---
    _f("inputs.starting.spouse_a_pretax_401k",
       "Spouse A pretax 401(k)", "number",
       group="Starting balances", tier="simple", step=1000,
       help="Current pretax 401(k) balance for Spouse A in today's dollars. "
            "Subject to RMDs starting at the RMD start age."),
    _f("inputs.starting.spouse_b_pretax_401k",
       "Spouse B pretax 401(k)", "number",
       group="Starting balances", tier="simple", step=1000,
       couple_only=True,
       help="Current pretax 401(k) balance for Spouse B in today's dollars. "
            "Subject to RMDs starting at the RMD start age."),
    _f("inputs.starting.spouse_a_pretax_ira",
       "Spouse A pretax IRA", "number",
       group="Starting balances", tier="simple", step=1000,
       help="Current pretax (traditional) IRA balance for Spouse A. Aggregated "
            "with the pretax 401(k) for Roth-conversion and RMD purposes."),
    _f("inputs.starting.spouse_b_pretax_ira",
       "Spouse B pretax IRA", "number",
       group="Starting balances", tier="simple", step=1000,
       couple_only=True,
       help="Current pretax (traditional) IRA balance for Spouse B. Aggregated "
            "with the pretax 401(k) for Roth-conversion and RMD purposes."),
    _f("inputs.starting.spouse_a_roth_ira",
       "Spouse A Roth IRA", "number",
       group="Starting balances", tier="simple", step=1000,
       help="Current Roth IRA balance for Spouse A. Withdrawals are tax-free "
            "after age 59½ and the 5-year seasoning rule."),
    _f("inputs.starting.spouse_b_roth_ira",
       "Spouse B Roth IRA", "number",
       group="Starting balances", tier="simple", step=1000,
       couple_only=True,
       help="Current Roth IRA balance for Spouse B. Withdrawals are tax-free "
            "after age 59½ and the 5-year seasoning rule."),
    _f("inputs.starting.hsa", "HSA balance", "number",
       group="Starting balances", tier="simple", step=1000,
       help="Current HSA balance. Tax-free for medical expenses; after age 65 "
            "behaves like a traditional IRA for non-medical withdrawals."),
    _f("inputs.starting.taxable_brokerage",
       "Taxable brokerage", "number",
       group="Starting balances", tier="simple", step=1000,
       help="Current taxable brokerage balance. Yields are taxed annually; "
            "sales create capital gains based on the cost-basis fraction."),
    _f("inputs.starting.pension_balance",
       "Pension cash balance", "number",
       group="Starting balances", tier="simple", step=1000,
       help="Current cash-balance value of the pension account. Grows at the "
            "plan's interest credit rate until annuitized at NRD."),

    # --- Household: Social Security (Simple) ---
    _f("inputs.ss.monthly_spouse_a", "Spouse A PIA at FRA ($/mo)",
       "number", group="Social Security", tier="simple", step=50,
       help="Spouse A's monthly Primary Insurance Amount at Full Retirement "
            "Age, in today's dollars (read off your SSA statement)."),
    _f("inputs.ss.monthly_spouse_b", "Spouse B PIA at FRA ($/mo)",
       "number", group="Social Security", tier="simple", step=50,
       couple_only=True,
       help="Spouse B's monthly Primary Insurance Amount at Full Retirement "
            "Age, in today's dollars (read off your SSA statement)."),
    _f("inputs.ss.start_age_a", "Spouse A claim age", "int",
       group="Social Security", tier="simple", min=62, max=70,
       help="Age at which Spouse A files for SS. 62=earliest (25-30% reduction); "
            "70=max delayed-credit boost (32% over FRA)."),
    _f("inputs.ss.start_age_b", "Spouse B claim age", "int",
       group="Social Security", tier="simple", min=62, max=70,
       couple_only=True,
       help="Age at which Spouse B files for SS. 62=earliest (25-30% reduction); "
            "70=max delayed-credit boost (32% over FRA)."),

    # --- Household: pension (Simple) ---
    _f("inputs.pension.balance_today", "Pension balance today",
       "number", group="Pension", tier="simple", step=1000,
       help="Current cash-balance value of your pension account. Grows by the "
            "plan's interest credit rate until you start drawing."),
    _f("inputs.pension.monthly_at_nrd", "Pension annuity at NRD ($/mo)",
       "number", group="Pension", tier="simple", step=50,
       help="Projected monthly pension benefit at Normal Retirement Date in "
            "today's dollars. Used in lieu of cash-balance once payments start."),
    _f("inputs.pension.start_age", "Pension start age (NRD)", "int",
       group="Pension", tier="simple", min=50, max=80,
       help="Age at which the pension starts paying. Often called Normal "
            "Retirement Date (NRD); typically 65 for most plans."),

    # --- Macro / horizon (Simple) ---
    _f("config.start_year", "Start year", "int",
       group="Macro & horizon", tier="simple", min=2020, max=2050,
       help="Calendar year the simulation begins. Anchors the federal tax "
            "regime, IRS contribution caps, and IRMAA brackets."),
    _f("config.horizon_age", "Horizon age", "int",
       group="Macro & horizon", tier="simple", min=70, max=110,
       help="Age of the *older* spouse when the simulation ends. Longer "
            "horizons give the optimizer more conversion runway."),
    _f("config.annual_expenses_today", "Annual expenses today",
       "number", group="Macro & horizon", tier="simple", step=1000,
       help="Household spending baseline in today's dollars. Inflated each "
            "year by `Spending inflation` (or main inflation if unset)."),
    _f("config.inflation", "Inflation", "percent",
       group="Macro & horizon", tier="simple",
       help="Annual CPI-style inflation. Decimal: 0.025 = 2.5%. Drives "
            "spending, SS COLA, and bracket indexing unless overridden."),
    _f("config.nominal_growth_rate",
       "Nominal growth rate (deterministic fallback)", "percent",
       group="Macro & horizon", tier="simple",
       help="Constant pre-tax portfolio growth rate used when the market "
            "model is `deterministic`. Ignored for stochastic models."),
    _f("config.tax_regime", "Federal tax regime", "select",
       options=_TAX_REGIMES,
       group="Macro & horizon", tier="simple",
       help="Federal bracket schedule. TCJA stays at current law; sunset "
            "reverts to the 2017 brackets after the 2025 sunset cliff."),
    _f("config.state_regime", "State tax regime", "select",
       options=_STATE_REGIMES,
       group="Macro & horizon", tier="simple",
       help="State income tax model applied to ordinary income, withdrawals, "
            "and conversions. Use 'Stateless' for FL/TX/WA/NV residents."),

    # --- Macro / horizon (Advanced) ---
    _f("config.wage_growth", "Wage growth", "percent",
       group="Macro (advanced)", tier="advanced",
       help="Annual nominal growth of W-2 wages (decimal). Typically slightly "
            "above inflation; 0.03 = 3% is a reasonable default."),
    _f("config.taxable_drag", "Taxable account drag", "percent",
       group="Macro (advanced)", tier="advanced",
       help="Annual tax-drag applied to taxable brokerage growth (yields, "
            "turnover). 0.005 = 50 bps is typical for a buy-and-hold ETF."),
    _f("config.ss_cola_rate",
       "SS COLA rate (blank = follow inflation)", "percent",
       group="Macro (advanced)", tier="advanced",
       help="Override for SS cost-of-living adjustment. Leave blank to track "
            "the main inflation rate; set to test divergent scenarios."),
    _f("config.bracket_indexing_rate",
       "Bracket indexing rate (blank = inflation)", "percent",
       group="Macro (advanced)", tier="advanced",
       help="Annual federal bracket indexing rate. Blank = main inflation. "
            "Useful for stress-testing 'brackets fall behind inflation'."),
    _f("config.regime_change_year_offset",
       "Regime swap year offset (blank = none)", "int",
       group="Macro (advanced)", tier="advanced", min=0, max=60,
       help="Years from start when the federal regime swaps (e.g. 2 = swap in "
            "year 2026 if start_year=2024). Blank = no mid-run swap."),
    _f("config.regime_change_target", "Regime swap target", "select",
       options=_REGIMES_NULLABLE,
       group="Macro (advanced)", tier="advanced",
       help="Federal regime to switch into at the offset above. Pair with "
            "'sunset' for the 2026 TCJA sunset model."),
    _f("config.state_regime_change_year_offset",
       "State regime swap year offset (blank = none)", "int",
       group="Macro (advanced)", tier="advanced", min=0, max=60,
       help="Years from start when the *state* regime swaps. Useful for "
            "modeling a planned move to a tax-free state mid-retirement."),
    _f("config.state_regime_change_target", "State regime swap target",
       "select", options=_STATE_REGIMES_NULLABLE,
       group="Macro (advanced)", tier="advanced",
       help="State regime to switch into at the offset above."),
    _f("config.section125_reduces_fica_wages",
       "S125 reduces FICA wages", "bool",
       group="Macro (advanced)", tier="advanced",
       help="If on, Section 125 cafeteria deductions (HSA, premiums) reduce "
            "the FICA wage base. Matches IRS treatment for most employer plans."),

    # --- Withdrawal & conversion strategy (Advanced) ---
    _f("config.withdrawal_strategy", "Withdrawal strategy", "select",
       options=_WITHDRAWAL_STRATEGIES,
       group="Withdrawal & conversion", tier="advanced",
       help="Order in which accounts are drained to meet spending. "
            "Bracket-fill blends conversions and withdrawals to hit a target bracket."),
    _f("config.bracket_fill_target", "Bracket-fill target",
       "percent", group="Withdrawal & conversion", tier="advanced",
       help="When strategy=bracket_fill, ordinary income is filled up to the "
            "top of this federal bracket. 0.22 = top of the 22% bracket."),
    _f("config.roth_conversion_target_bracket",
       "Roth conversion target bracket", "percent",
       group="Withdrawal & conversion", tier="advanced",
       help="Top of the bracket the optimizer fills with annual Roth "
            "conversions. 0.0 = no conversions; 0.32 = fill the 32% bracket."),
    _f("config.roth_conversion_amount",
       "Fixed Roth conversion amount", "number",
       group="Withdrawal & conversion", tier="advanced", step=1000,
       help="Override: convert this fixed dollar amount per year (today's $) "
            "instead of using the bracket-fill rule. 0 disables the override."),
    _f("config.cap_conversion_by_liquidity",
       "Cap conversion by liquidity", "bool",
       group="Withdrawal & conversion", tier="advanced",
       help="If on, conversions are capped at the cash available to pay the "
            "conversion tax (so the user never has to dip into pretax to fund tax)."),
    _f("config.conversion_taxable_use_ratio",
       "Conversion taxable-use ratio", "percent",
       group="Withdrawal & conversion", tier="advanced",
       help="Fraction of conversion tax paid from the taxable brokerage. "
            "Rest comes from pretax (which itself triggers more tax)."),
    _f("config.protect_roth_in_conversion_years",
       "Protect Roth in conversion years", "bool",
       group="Withdrawal & conversion", tier="advanced",
       help="If on, the withdrawal cascade skips Roth in years where a "
            "conversion happened, preserving the Roth's tax-free growth."),
    _f("config.rmd_start_age", "RMD start age", "int",
       group="Withdrawal & conversion", tier="advanced", min=72, max=80,
       help="Age at which Required Minimum Distributions begin. Current law: "
            "73 (birth years 1951-1959); 75 (1960+)."),
    _f("config.cap_gains_basis_fraction",
       "Capital gains cost-basis fraction", "percent",
       group="Withdrawal & conversion", tier="advanced",
       help="Fraction of the taxable brokerage that is cost basis. 0.6 = 60% "
            "basis, 40% unrealized gain. Drives LTCG on sales."),
    _f("config.heir_marginal_rate", "Heir marginal rate", "percent",
       group="Withdrawal & conversion", tier="advanced",
       help="Marginal tax rate the heirs pay on inherited *pretax* balances. "
            "Used to net-down terminal NW for the action plan's headline."),
    _f("config.optimize_ss_claim_age",
       "Optimize SS claim age (S3)", "bool",
       group="Withdrawal & conversion", tier="advanced",
       help="If on, the optimizer (S3) treats SS claim age (62-70) as a "
            "decision axis. Otherwise the user-set ages are fixed."),

    # --- Healthcare & Medicare (Advanced) ---
    _f("config.medicare_base_b_d_premium",
       "Medicare Part B+D base premium / yr", "number",
       group="Healthcare & Medicare", tier="advanced", step=100,
       help="Base annual Medicare Part B + Part D premium per person, before "
            "IRMAA surcharges. ~$2,200/yr is the 2024 baseline."),
    _f("config.health_pre65_today", "Pre-Medicare health cost / yr",
       "number", group="Healthcare & Medicare", tier="advanced", step=100,
       help="Annual out-of-pocket health cost per adult before age 65 (today's $). "
            "Reduced by ACA subsidies if ACA is enabled."),
    _f("config.irmaa_lookback_years", "IRMAA lookback years", "int",
       group="Healthcare & Medicare", tier="advanced", min=0, max=3,
       help="Years between MAGI and the IRMAA tier it triggers. Current rule: 2 "
            "(your 2024 MAGI sets your 2026 IRMAA tier)."),
    _f("config.aca_enabled", "ACA premium tax credit enabled", "bool",
       group="Healthcare & Medicare", tier="advanced",
       help="If on, MAGI under ACA thresholds qualifies for the Premium Tax "
            "Credit, which offsets `Pre-Medicare health cost`."),
    _f("config.aca_benchmark_premium_per_adult",
       "ACA benchmark premium / adult / yr", "number",
       group="Healthcare & Medicare", tier="advanced", step=500,
       help="Second-lowest-cost silver plan premium per adult per year (today's $). "
            "Anchors the ACA subsidy formula."),
    _f("config.aca_max_contrib_pct", "ACA max contrib % of MAGI",
       "percent", group="Healthcare & Medicare", tier="advanced",
       help="Cap on premium contribution as a fraction of MAGI. 0.085 = 8.5% "
            "(the post-IRA cap; pre-IRA was higher)."),
    _f("config.stepup_at_first_death",
       "Step-up basis at first death", "bool",
       group="Healthcare & Medicare", tier="advanced",
       help="If on, the taxable brokerage basis steps up to FMV when the first "
            "spouse dies. Eliminates the unrealized gain on community-property assets."),

    # --- Taxable yields (Advanced) ---
    _f("config.taxable_equity_div_yield",
       "Taxable equity dividend yield", "percent",
       group="Taxable yields", tier="advanced",
       help="Annual dividend yield on the taxable equity sleeve. 0.018 = "
            "1.8% is roughly an S&P 500 ETF."),
    _f("config.taxable_bond_interest_yield",
       "Taxable bond interest yield", "percent",
       group="Taxable yields", tier="advanced",
       help="Annual coupon yield on the taxable bond sleeve. Taxed as ordinary "
            "income each year. 0.04 = 4% for current Treasuries."),
    _f("config.taxable_equity_qualified_fraction",
       "Qualified-dividend fraction", "percent",
       group="Taxable yields", tier="advanced",
       help="Fraction of equity dividends that meet the IRS qualified-dividend "
            "test (and are taxed at LTCG rates). Rest is ordinary."),

    # --- Mortality (Advanced) ---
    _f("config.mortality.year_of_death_a",
       "Year of death - Spouse A", "int",
       group="Mortality", tier="advanced", min=1, max=80,
       help="Year offset (from start) when Spouse A dies. Triggers MFJ→single "
            "filing transition and spousal account rollover. Set high to disable."),
    _f("config.mortality.year_of_death_b",
       "Year of death - Spouse B", "int",
       group="Mortality", tier="advanced", min=1, max=80,
       couple_only=True,
       help="Year offset (from start) when Spouse B dies. Triggers MFJ→single "
            "filing transition and spousal account rollover. Set high to disable."),
    _f("config.mortality.pension_survivor_pct",
       "Pension survivor %", "percent",
       group="Mortality", tier="advanced",
       help="Fraction of pension benefit retained by the survivor under a "
            "joint-and-survivor annuity. 0.50 = 50% J&S; 1.00 = 100% J&S."),
    _f("config.mortality.ss_survivor_keeps_higher",
       "SS survivor keeps higher PIA", "bool",
       group="Mortality", tier="advanced",
       help="If on, the surviving spouse keeps the higher of the two SS "
            "benefits and forfeits the lower one (per SSA rules)."),

    # --- Market model (Advanced) ---
    _f("config.market.kind", "Market model", "select",
       options=_MARKET_KINDS,
       group="Market model", tier="advanced",
       help="Asset return generator. Lognormal = stochastic; deterministic = "
            "constant returns; bootstrap / historical_sequence = empirical."),
    # Lognormal block
    _f("config.market.equity_mu", "Equity mu", "percent",
       group="Market model", tier="advanced",
       help="Lognormal arithmetic mean of annual equity returns. 0.07 = 7%/yr "
            "before vol drag (lognormal only)."),
    _f("config.market.equity_sigma", "Equity sigma", "percent",
       group="Market model", tier="advanced",
       help="Lognormal volatility (std dev) of annual equity returns. 0.18 = "
            "18%/yr is a long-run S&P estimate (lognormal only)."),
    _f("config.market.bond_mu", "Bond mu", "percent",
       group="Market model", tier="advanced",
       help="Lognormal arithmetic mean of annual bond returns. 0.04 = 4%/yr "
            "for an aggregate bond fund (lognormal only)."),
    _f("config.market.bond_sigma", "Bond sigma", "percent",
       group="Market model", tier="advanced",
       help="Lognormal volatility of annual bond returns. 0.06 = 6%/yr is "
            "typical for an aggregate bond fund (lognormal only)."),
    _f("config.market.equity_bond_corr", "Equity-bond correlation",
       "number", group="Market model", tier="advanced", step=0.01,
       help="Correlation between equity and bond returns. Range -1 to 1; "
            "0.0 = independent; -0.2 is a long-run estimate."),
    _f("config.market.cape_today",
       "CAPE today (blank = no scaling)", "number",
       group="Market model", tier="advanced", step=0.5,
       help="Current Shiller CAPE ratio. If set, equity_mu scales by "
            "long-run/today (lower mu when CAPE is high). Blank = no scaling."),
    _f("config.market.cape_long_run", "CAPE long-run mean",
       "number", group="Market model", tier="advanced", step=0.5,
       help="Long-run Shiller CAPE used as the anchor for valuation-based mu "
            "scaling. ~17 is the 1881-2024 average."),
    # Deterministic block
    _f("config.market.equity",
       "Deterministic equity return (only if kind=deterministic)",
       "percent", group="Market model", tier="advanced",
       help="Constant annual equity return when kind=deterministic. Ignored "
            "for all stochastic market models."),
    _f("config.market.bond",
       "Deterministic bond return (only if kind=deterministic)",
       "percent", group="Market model", tier="advanced",
       help="Constant annual bond return when kind=deterministic. Ignored "
            "for all stochastic market models."),
    # Bootstrap block
    _f("config.market.block_size",
       "Bootstrap block size (only if kind=bootstrap)", "int",
       group="Market model", tier="advanced", min=1, max=20,
       help="Block length for the block-bootstrap of historical returns. "
            "Larger blocks preserve serial correlation. 5-10 is typical."),

    # --- Asset location (Advanced) ---
    _f("config.asset_location.pretax_equity_pct",
       "Pretax equity %", "percent",
       group="Asset location", tier="advanced",
       help="Equity allocation in the pretax 401(k)/IRA. Rest is bonds. "
            "Most plans hold the bulk of bonds here for tax-efficiency."),
    _f("config.asset_location.roth_equity_pct",
       "Roth equity %", "percent",
       group="Asset location", tier="advanced",
       help="Equity allocation in Roth accounts. Tax-free growth makes Roth "
            "the ideal home for high-expected-return assets."),
    _f("config.asset_location.taxable_equity_pct",
       "Taxable equity %", "percent",
       group="Asset location", tier="advanced",
       help="Equity allocation in the taxable brokerage. Equities are "
            "tax-efficient (LTCG rates + qualified dividends)."),
    _f("config.asset_location.hsa_equity_pct",
       "HSA equity %", "percent",
       group="Asset location", tier="advanced",
       help="Equity allocation in the HSA. HSA is the only triple-tax-advantaged "
            "account, so usually 100% equities for max long-run growth."),

    # --- Spending profile (Advanced) ---
    _f("config.spending.kind", "Spending profile", "select",
       options=_SPENDING_KINDS,
       group="Spending profile", tier="advanced",
       help="Flat = constant real spending across all years. Smile = the "
            "go-go/slow-go/no-go retirement smile (Blanchett-style)."),
    _f("config.spending.base_spending",
       "Base spending (today $)", "number",
       group="Spending profile", tier="advanced", step=1000,
       help="Annual real spending baseline before age modifiers. Inflated "
            "each year and scaled by the smile if profile=smile."),
    _f("config.spending.inflation",
       "Spending inflation", "percent",
       group="Spending profile", tier="advanced",
       help="Override annual inflation applied to spending. Blank = follow "
            "the main inflation rate."),
    _f("config.spending.ltc_years",
       "LTC shock years (smile only)", "int",
       group="Spending profile", tier="advanced", min=0, max=15,
       help="Number of end-of-life years modeled with elevated long-term-care "
            "spending. 0 disables the LTC shock. Smile profile only."),
    _f("config.spending.ltc_annual_today",
       "LTC annual today $ (smile only)", "number",
       group="Spending profile", tier="advanced", step=1000,
       help="Annual LTC cost during the LTC shock years (today's $). National "
            "median is roughly $100k-$120k/yr for nursing care."),

    # --- IRA / Mega-backdoor (Advanced) ---
    _f("inputs.spouse_a_after_tax_401k_pct",
       "Spouse A after-tax 401(k) %", "percent",
       group="IRA & Mega-backdoor", tier="advanced",
       help="Fraction of Spouse A's gross routed to *after-tax* 401(k) (the "
            "mega-backdoor source). Only if your plan supports it."),
    _f("inputs.spouse_b_after_tax_401k_pct",
       "Spouse B after-tax 401(k) %", "percent",
       group="IRA & Mega-backdoor", tier="advanced",
       couple_only=True,
       help="Fraction of Spouse B's gross routed to *after-tax* 401(k) (the "
            "mega-backdoor source). Only if your plan supports it."),
    _f("inputs.spouse_a_mega_backdoor_enabled",
       "Spouse A mega-backdoor enabled", "bool",
       group="IRA & Mega-backdoor", tier="advanced",
       help="If on, Spouse A's after-tax 401(k) contributions are converted "
            "to Roth in-plan annually (the 'mega-backdoor Roth' move)."),
    _f("inputs.spouse_b_mega_backdoor_enabled",
       "Spouse B mega-backdoor enabled", "bool",
       group="IRA & Mega-backdoor", tier="advanced",
       couple_only=True,
       help="If on, Spouse B's after-tax 401(k) contributions are converted "
            "to Roth in-plan annually (the 'mega-backdoor Roth' move)."),
    _f("inputs.spouse_a_traditional_ira_contrib",
       "Spouse A Trad IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500,
       help="Annual traditional IRA contribution for Spouse A (today's $). "
            "Subject to deductibility phase-outs based on income."),
    _f("inputs.spouse_b_traditional_ira_contrib",
       "Spouse B Trad IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500,
       couple_only=True,
       help="Annual traditional IRA contribution for Spouse B (today's $). "
            "Subject to deductibility phase-outs based on income."),
    _f("inputs.spouse_a_roth_ira_contrib",
       "Spouse A direct Roth IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500,
       help="Annual *direct* Roth IRA contribution for Spouse A. Phased out "
            "above MAGI thresholds; use the backdoor knob if over the cap."),
    _f("inputs.spouse_b_roth_ira_contrib",
       "Spouse B direct Roth IRA contribution", "number",
       group="IRA & Mega-backdoor", tier="advanced", step=500,
       couple_only=True,
       help="Annual *direct* Roth IRA contribution for Spouse B. Phased out "
            "above MAGI thresholds; use the backdoor knob if over the cap."),
    _f("inputs.spouse_a_backdoor_roth",
       "Spouse A backdoor Roth", "bool",
       group="IRA & Mega-backdoor", tier="advanced",
       help="If on, contribute to a non-deductible IRA and convert to Roth "
            "when MAGI is too high for direct Roth contributions."),
    _f("inputs.spouse_b_backdoor_roth",
       "Spouse B backdoor Roth", "bool",
       group="IRA & Mega-backdoor", tier="advanced",
       couple_only=True,
       help="If on, contribute to a non-deductible IRA and convert to Roth "
            "when MAGI is too high for direct Roth contributions."),

    # --- Social Security (Advanced) ---
    _f("inputs.ss.fra_a", "Spouse A FRA", "int",
       group="Social Security (advanced)", tier="advanced", min=62, max=70,
       help="Spouse A's Full Retirement Age. 67 for those born 1960+; "
            "between 66 and 67 for earlier birth years."),
    _f("inputs.ss.fra_b", "Spouse B FRA", "int",
       group="Social Security (advanced)", tier="advanced", min=62, max=70,
       couple_only=True,
       help="Spouse B's Full Retirement Age. 67 for those born 1960+; "
            "between 66 and 67 for earlier birth years."),

    # --- Pension (Advanced) ---
    _f("inputs.pension.years_of_service_today",
       "Years of service today", "int",
       group="Pension (advanced)", tier="advanced", min=0, max=60,
       help="Pension service years already credited as of the start year. "
            "Drives the cash-balance accrual rate and benefit projections."),
    _f("inputs.pension.pre_2016_participant",
       "Pre-2016 participant (5% floor)", "bool",
       group="Pension (advanced)", tier="advanced",
       help="If on, the pension uses the pre-2016 5% interest credit floor "
            "instead of the 30-yr Treasury rate. Common for legacy plans."),
    _f("inputs.pension.interest_rate",
       "Interest rate (blank = floor)", "percent",
       group="Pension (advanced)", tier="advanced",
       help="Annual cash-balance interest credit rate. Blank = use the plan's "
            "floor (5% for pre-2016 participants, 30-yr Treasury otherwise)."),
    _f("inputs.pension.irs_comp_limit_today",
       "IRS comp limit today $", "number",
       group="Pension (advanced)", tier="advanced", step=1000,
       help="IRS §415(c) annual compensation limit for the start year. "
            "$345k for 2024. Caps pension-eligible salary."),

    # --- Health premiums (Advanced) ---
    _f("inputs.health_premiums.spouse_a_medical",
       "Spouse A medical premium", "number",
       group="Health premiums", tier="advanced", step=100,
       help="Annual *post-tax* medical premium portion for Spouse A (today's $). "
            "Pretax/Section 125 premiums should be excluded here."),
    _f("inputs.health_premiums.spouse_a_dental",
       "Spouse A dental premium", "number",
       group="Health premiums", tier="advanced", step=50,
       help="Annual dental premium for Spouse A (today's $). Typically a "
            "small post-tax line item."),
    _f("inputs.health_premiums.spouse_a_vision",
       "Spouse A vision premium", "number",
       group="Health premiums", tier="advanced", step=50,
       help="Annual vision premium for Spouse A (today's $)."),
    _f("inputs.health_premiums.spouse_b_medical",
       "Spouse B medical premium", "number",
       group="Health premiums", tier="advanced", step=100,
       couple_only=True,
       help="Annual *post-tax* medical premium portion for Spouse B (today's $). "
            "Pretax/Section 125 premiums should be excluded here."),
    _f("inputs.health_premiums.spouse_b_dental",
       "Spouse B dental premium", "number",
       group="Health premiums", tier="advanced", step=50,
       couple_only=True,
       help="Annual dental premium for Spouse B (today's $). Typically a "
            "small post-tax line item."),
    _f("inputs.health_premiums.spouse_b_vision",
       "Spouse B vision premium", "number",
       group="Health premiums", tier="advanced", step=50,
       couple_only=True,
       help="Annual vision premium for Spouse B (today's $)."),
)


def fields_by_tier(tier: str) -> tuple[FormField, ...]:
    """Return the form fields belonging to `tier` ('simple' or 'advanced')."""
    return tuple(f for f in FIELD_SCHEMA if f.tier == tier)


def fields_by_group(fields: Iterable[FormField]) -> dict[str, list[FormField]]:
    """Group fields by their `group` for sectioned form rendering."""
    out: dict[str, list[FormField]] = {}
    for f in fields:
        out.setdefault(f.group, []).append(f)
    return out


_FIELDS_BY_PATH: dict[str, FormField] = {f.path: f for f in FIELD_SCHEMA}


def get_field(path: str) -> FormField | None:
    return _FIELDS_BY_PATH.get(path)


def all_paths() -> tuple[str, ...]:
    return tuple(f.path for f in FIELD_SCHEMA)
