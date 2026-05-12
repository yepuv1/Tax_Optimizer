# Scenarios

This directory holds JSON scenario files consumed by `tax_optimizer`.
A scenario is just a `{ "config": {...}, "inputs": {...} }` dict that
overrides defaults on `Config` / `Inputs`. The CLI loads them with
`--scenario PATH`, and the library API uses `apply_scenario(cfg,
inputs, scenario_dict)`.

## Files

| File              | Purpose                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `template.json`   | **Canonical reference**: every available knob, with its default value. Copy this to start a new plan. |
| `example01.json`  | Mid-career dual-income household with a widow stress test.                                            |
| `example02.json`  | Late-career household with mega-backdoor Roth, pension, and aggressive savings.                       |

Looking specifically for the Roth-conversion or withdrawal-strategy
knobs? Jump to
[Roth conversion & withdrawal knobs](#roth-conversion--withdrawal-knobs).

## Maintenance contract for new knobs

**Whenever a new field is added to `Config`, `Inputs`, or one of the
nested input dataclasses (`StartingBalances`, `CurrentIncome`,
`CurrentContrib`, `PensionInputs`, `SocialSecurity`) or to `Mortality`,
the contributor MUST also:**

1. Add the field (with its default value) to `template.json`.
2. If the new knob is meaningful for the two narrative scenarios,
   reflect it in `example01.json` / `example02.json` too (else they
   silently fall back to defaults).
3. Re-run the drift tests:

   ```
   pytest tests/test_scenario_template.py
   ```

   These tests **will fail loudly** if `template.json` is missing a
   new dataclass field or has a value that disagrees with the
   field's default. The failure message points at the exact field
   and which side to update.

`scenarios/example02.local.json` is a developer's personal scenario,
listed in `.gitignore`. Do not commit personal financial data.

## Polymorphic blocks

`config.market` and `config.spending` are tagged-union shapes. The
template renders ONE representative form (lognormal + smile) so the
file remains a useful copy-paste starting point. Other shapes the
loader accepts:

### `config.market`

```jsonc
// kind: "deterministic" — constant returns
{ "kind": "deterministic", "equity": 0.07, "bond": 0.04 }

// kind: "lognormal" — independent yearly draws (default; CAPE-aware)
{ "kind": "lognormal", "equity_mu": 0.07, "equity_sigma": 0.18,
  "bond_mu": 0.04, "bond_sigma": 0.06, "equity_bond_corr": 0.10,
  "cape_today": null, "cape_long_run": 16.5 }

// kind: "lognormal" with a CMA preset shortcut
{ "kind": "lognormal", "cma": "vanguard_2025" }

// kind: "bootstrap" — block-bootstrap from history
{ "kind": "bootstrap", "block_size": 5 }

// kind: "historical_sequence" — replay contiguous N-year slices
{ "kind": "historical_sequence" }
```

### `config.spending`

```jsonc
// kind: "flat" — constant real spending
{ "kind": "flat", "base_spending": 85000, "inflation": 0.025 }

// kind: "smile" — Blanchett "retirement smile" + LTC shock at end of life
{ "kind": "smile", "base_spending": 85000, "inflation": 0.025,
  "ltc_years": 3, "ltc_annual_today": 80000 }

// kind: "custom" — arbitrary phases / lump events
{ "kind": "custom", "base_spending": 85000, "inflation": 0.025,
  "phases": [{ "age_lo": 65, "age_hi": 200, "multiplier": 1.0,
               "label": "retirement" }],
  "lump_events": [{ "year_offset": 5, "amount_today": 50000,
                    "label": "kitchen-remodel" }],
  "ltc_shock": { "years": 3, "annual_cost_today": 80000 } }
```

### `config.state_regime`

String code: `"stateless"` (default), `"CA"`, `"NY"`, `"IL"`, `"MA"`.

### `config.tax_regime` / `config.regime_change_target`

String code: `"tcja"` (alias `"tcja_extended"`), `"pre_tcja"` (alias
`"pre_tcja_2017"`), `"sunset"` (alias `"sunset_2026"`).

### `config.asset_location`

Either `{ "uniform_equity_pct": 0.7 }` (every account allocated the
same way) OR the per-bucket form shown in `template.json`.

## Roth conversion & withdrawal knobs

A focused reference for the two most-tuned strategy axes. All defaults
below match `template.json`; file references point at the canonical
definition.

### A. Roth-conversion sizing (`Config`)

#### A1. Pick a mode (exactly one)

| Knob                              | Default | What it does                                                                                                                                                                                                                                                  |
| --------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `roth_conversion_target_bracket`  | `0.0`   | **Bracket-fill mode.** Convert just enough to fill the named ordinary bracket (e.g. `0.22`, `0.24`, `0.32`). Self-regulates to zero in high-income working years; fires in any year with positive pretax balance.                                             |
| `roth_conversion_amount`          | `0.0`   | **Fixed-dollar mode.** Convert exactly this many of today's dollars per year, capped at available pretax minus RMD. Gated to the gap window (retired but pre-`rmd_start_age`) so it doesn't detonate working-year tax bills.                                  |

Set ONE of these `> 0` to enable conversions. If both are `0`, no
conversions happen. See `tax_optimizer/conversion.py`
(`planned_roth_conversion`).

#### A2. v6.5 liquidity guards (default-on)

| Knob                                  | Default | What it does                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cap_conversion_by_liquidity`         | `true`  | The simulator bisects the chosen conversion total DOWN if its marginal federal + state tax would exceed the household's tax-paying capacity (earned cash + pension + SS + RMD net of FICA/SDI/spending/healthcare/IRA & MBR contributions, plus the taxable-balance slice). Without this, an aggressive bracket target or oversized fixed amount silently raids the just-funded Roth.   |
| `conversion_taxable_use_ratio`        | `0.5`   | Fraction of the taxable-brokerage balance the sizer is allowed to earmark for conversion tax. `0.0` → tax must come entirely from earned/retirement-income surplus. `1.0` → use full taxable.                                                                                                                                                                                           |
| `protect_roth_in_conversion_years`    | `true`  | The deficit cascade refuses to touch Roth in any year a conversion fires. Liquidity overshoots surface as `unfunded` instead of silently draining Roth dollars.                                                                                                                                                                                                                         |

See `tax_optimizer/config.py` (Roth-conversion liquidity-guards
block).

#### A3. Diagnostic columns (read-only, every row)

- `roth_conversion`, `roth_conversion_a`, `roth_conversion_b` — what got converted.
- `roth_conv_capped_by_liquidity` — `True` when A2 bound and reduced the size.
- `roth_conv_bracket_target` — what the bracket / fixed mode wanted before the liquidity cap.
- `roth_conv_tax_capacity` — the per-year capacity number.

### B. Withdrawal strategy (`Config`)

#### B1. Strategy selection

| Knob                  | Default          | Values                                                  |
| --------------------- | ---------------- | ------------------------------------------------------- |
| `withdrawal_strategy` | `"conventional"` | `"conventional"` / `"proportional"` / `"bracket_fill"`  |

- **`conventional`** (Bengen / Schwab default): RMD first → taxable → pretax → Roth.
- **`proportional`**: pull from taxable / pretax-room / Roth in proportion to balances.
- **`bracket_fill`**: fill bracket headroom with pretax before touching taxable, then Roth as the residual.

See `tax_optimizer/withdrawals.py` (`withdraw_for_need`).

#### B2. Strategy parameters

| Knob                       | Default | What it does                                                                                                                                                       |
| -------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bracket_fill_target`      | `0.22`  | Bracket cap (as a decimal rate) the `bracket_fill` strategy fills pretax to. Also the implicit ceiling on incidental pretax draws in `proportional`.               |
| `rmd_start_age`            | `75`    | First age the IRS Uniform Lifetime Table is applied. Validated `>= 72`. Lowering to `73` models the current SECURE 2.0 rule.                                       |
| `cap_gains_basis_fraction` | `0.5`   | Initial cost-basis / FMV ratio for the taxable brokerage. The simulator tracks `cumulative_basis` live thereafter (this knob is just the seed).                    |

#### B3. Deficit cascade (no direct knob — the safety net under every strategy)

When `cash_inflow - taxes - net_need < 0`, `cover_deficit` pulls in
this order:

**taxable** (LTCG-rate gross-up) → **Roth** (skipped in conversion
years per A2) → **HSA after age 65** (ordinary gross-up, no penalty)
→ **pretax** (ordinary gross-up).

Anything left over shows up as `unfunded` on that row.

See `tax_optimizer/withdrawals.py` (`cover_deficit`).

### C. Knobs that indirectly shape both

| Knob                                                                                                                | Why it matters for conversion / withdrawal                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tax_regime` / `regime_change_year_offset` / `regime_change_target`                                                 | TCJA-extended vs. SUNSET shifts marginal rates ~3–4 pts above ~$100k AGI. A conversion that fills the 22% bracket today fills the 25% bracket post-sunset — bracket-fill math adjusts automatically once the regime swaps.                            |
| `state_regime` / `state_regime_change_*`                                                                            | High-tax states (CA, NY) add 9.3% / 6.85% marginal bite to conversion tax. The v6.5 capacity bisection includes state tax. CA → FL move post-retirement is a classic conversion-timing optimization.                                                  |
| `bracket_indexing_rate`                                                                                             | Brackets / std-deduction / IRMAA tiers index by this rate. `0.0` freezes them (worst-case "Congress freezes brackets" stress).                                                                                                                        |
| `taxable_equity_div_yield`, `taxable_bond_interest_yield`, `taxable_equity_qualified_fraction`                      | Portfolio yields hit AGI every year and eat into bracket headroom available for conversions.                                                                                                                                                          |
| `medicare_base_b_d_premium`, `health_pre65_today`, `irmaa_lookback_years`, `aca_enabled` / `aca_*`                  | Healthcare costs are subtracted from `tax_paying_capacity`. IRMAA tier hops are a real cost of aggressive same-year conversions (2-year lookback means they show up later).                                                                           |
| `heir_marginal_rate`                                                                                                | The optimizer weights pretax → Roth conversions against this rate. Higher heir rate (32%+) → more aggressive conversions. `0.0` → roughly indifferent.                                                                                                |
| `stepup_at_first_death`                                                                                             | If `true`, taxable basis resets at first spouse's death, making "leave it taxable" cheaper for heirs → mildly de-emphasizes conversions.                                                                                                              |
| `mortality.year_of_death_a/b` + `mortality.pension_survivor_pct`                                                    | Conversion timing changes when one spouse dies (filing status flips to single, brackets compress) — sized automatically against the active regime.                                                                                                    |
| `optimize_ss_claim_age`                                                                                             | When `true`, the optimizer also chooses SS claim ages; later claim → longer gap window with low ordinary income → more bracket headroom for conversions.                                                                                              |

### D. Inputs that gate conversion / withdrawal timing

| Field (on `Inputs` or nested)                            | Effect                                                                                                                                                            |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spouse_a_retire_age` / `spouse_b_retire_age`            | Fixed-amount conversions only fire when the eligible spouse is retired AND pre-RMD. Bracket-fill ignores this gate.                                               |
| `ss.start_age` / `ss.start_age_a` / `ss.start_age_b`     | Claim age changes the bracket headroom available in the gap. Later claim = wider conversion window.                                                               |
| `pension.start_age`                                      | Pension annuity income reduces bracket headroom once it starts.                                                                                                   |
| `starting.spouse_*_pretax_*`                             | Hard cap on conversions (each spouse's conversion is capped at their pretax balance minus their RMD).                                                             |
| `starting.spouse_*_pretax_ira`                           | The backdoor pro-rata rule (IRC §408(d)(2)) taxes the Roth-conversion piece pro-rata against the pretax-IRA sub-balance. Tracked separately from pretax-401(k).   |

### E. Common patterns

```jsonc
// Aggressive gap-year conversions in a CA-then-FL household
"config": {
  "roth_conversion_target_bracket": 0.24,
  "cap_conversion_by_liquidity": true,
  "conversion_taxable_use_ratio": 0.75,    // willing to spend taxable on conv tax
  "state_regime": "CA",
  "state_regime_change_year_offset": 11,   // year retirement starts
  "state_regime_change_target": "stateless"
}

// Pure bracket-fill withdrawal, no Roth conversions
"config": {
  "withdrawal_strategy": "bracket_fill",
  "bracket_fill_target": 0.22,
  "roth_conversion_target_bracket": 0.0
}

// Fixed $50k/yr gap-year conversions, conservative liquidity stance
"config": {
  "roth_conversion_amount": 50000.0,
  "cap_conversion_by_liquidity": true,
  "conversion_taxable_use_ratio": 0.0,    // earned-income surplus only
  "protect_roth_in_conversion_years": true
}
```

## Useful CLI patterns

Generate a fresh defaults dump (newer than `template.json` if you've
been editing on a feature branch):

```
tax-optimizer --print-defaults > /tmp/defaults.json
```

Run a scenario:

```
tax-optimizer --scenario scenarios/example01.json
```

Override a single value without editing the file:

```
tax-optimizer --scenario scenarios/example01.json \
  --set config.roth_conversion_target_bracket=0.22
```
