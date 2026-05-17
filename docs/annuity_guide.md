# Annuity Configuration Guide

How to model an annuity contract — and the new pension/annuity
**lump-sum** election — in Tax Optimizer. Three configuration paths
are covered here, in order of casual → power-user:

1. The **Dash app** (form-driven, point-and-click).
2. **Scenario JSON files** (saved, sharable, version-controllable).
3. The **CLI** (`tax-optimizer ...`) and the Python API.

If you only want a quick reference for what each knob *does*, jump
straight to [Knob reference](#knob-reference). For the math behind
the §72(b) exclusion ratio and a year-by-year worked example, see
the deep-dive in `CHANGELOG.md` under *"Annuity account type +
lump-sum knob"*.

---

## Contents

- [What this models](#what-this-models)
- [The three lump-sum modes](#the-three-lump-sum-modes)
- [Configuring from the Dash app](#configuring-from-the-dash-app)
  - [Quickstart: a $200k qualified annuity](#quickstart-a-200k-qualified-annuity)
  - [Modeling a non-qualified contract](#modeling-a-non-qualified-contract)
  - [Electing a pension lump-sum](#electing-a-pension-lump-sum)
- [Configuring via scenario JSON](#configuring-via-scenario-json)
  - [Minimal annuity block](#minimal-annuity-block)
  - [Pension lump-sum election](#pension-lump-sum-election)
  - [Two ready-to-run examples](#two-ready-to-run-examples)
  - [Loading a custom scenario](#loading-a-custom-scenario)
- [Configuring from the CLI / Python API](#configuring-from-the-cli--python-api)
  - [`--set` overrides](#set-overrides)
  - [Python: building `Inputs` programmatically](#python-building-inputs-programmatically)
- [Knob reference](#knob-reference)
- [Validation: errors you might hit](#validation-errors-you-might-hit)
- [Reading the output](#reading-the-output)
- [FAQ / common questions](#faq--common-questions)

---

## What this models

Two related additions, both shipped together:

1. **A new annuity account type** (`inputs.annuity`) — a contract you
   already own that pays a fixed monthly amount starting at
   `start_age`. It can be either:
   - **Qualified** — funded with pretax dollars; every payout dollar
     is ordinary income. Behaves like a small private pension.
   - **Non-qualified** — funded with after-tax dollars and tracks a
     `cost_basis`. The IRS §72(b) **exclusion ratio** splits each
     payment into a tax-free return-of-basis portion and a taxable
     gain portion, until the basis is recovered.

2. **A `lump_sum_mode` knob on BOTH the pension and the annuity**.
   At `start_age`, instead of starting monthly payments, you can
   elect to take the contract as a single lump sum — either rolled
   into a pretax IRA tax-free, or distributed in cash with full
   ordinary-income tax (and the 10% early-distribution surtax if
   under 59½).

Pensions are always qualified plans, so they don't have a
`tax_kind` knob — only `lump_sum_mode`.

---

## The three lump-sum modes

The same three options work for both `inputs.pension.lump_sum_mode`
and `inputs.annuity.lump_sum_mode`:

| Mode | What happens at `start_age` | Tax that year |
|------|------------------------------|---------------|
| `"none"` *(default)* | Monthly payments begin as before. Existing scenarios behave identically — backward compatible. | Each payment taxed at marginal ordinary rates. Non-qualified annuity payments split via the exclusion ratio. |
| `"rollover_pretax"` | Whole balance moves into the participant's pretax IRA (`state.spouse_a_pretax_ira`) per IRC §402(c). | **$0.** Direct rollover is tax-free. Future RMDs and ordinary withdrawals apply automatically via the existing pretax pipeline. |
| `"cash"` | Whole balance distributed in a single year. Net cash lands in `state.taxable`. | **Full balance** ordinary income (qualified) or **balance − cost_basis** ordinary income (non-qualified). Plus a **10% additional tax** under IRC §72(t) / §72(q) if `a_age < 60`. |

> **Forbidden combination:** `tax_kind="non_qualified"` with
> `lump_sum_mode="rollover_pretax"` raises a `ValueError` at
> `Inputs(...)` construction time. The IRS prohibits rolling a
> non-qualified annuity into a qualified IRA (IRC §408 / §402(c)).
> Use `"cash"` (taxable surrender) instead.

---

## Configuring from the Dash app

Launch the dashboard:

```bash
tax-optimizer-app
# → http://127.0.0.1:8050
```

The annuity knobs live in two new form groups:

- **"Annuity"** — Simple-tier group with the four most-common
  fields. Visible by default.
- **"Annuity (advanced)"** — Advanced-tier group with the §72(b)
  exclusion-ratio knobs and the pre-payout growth rate. Visible
  only when you toggle the **"Show advanced"** switch in the
  sidebar.

The pension lump-sum selector lives in the existing **"Pension"**
group, right after `Pension start age (NRD)`.

Every field carries a hover tooltip with a one-paragraph
explanation; this guide is the long-form version.

### Quickstart: a $200k qualified annuity

Goal: model a $200k qualified annuity that starts paying $1,000/mo
at age 65, taxed entirely as ordinary income.

1. Open the dashboard. Find the **"Annuity"** group in the sidebar
   (right after **"Pension"**).
2. Set **Annuity balance today** to `200000`.
3. Set **Annuity monthly payment ($/mo)** to `1000`.
4. Set **Annuity start age** to `65`.
5. Leave **Annuity lump-sum election** as `"None (monthly payments)"`.
6. Click **Run** at the top of the sidebar.

In the **Year-by-year** tab you'll see new columns:
`annuity_taxable` ($12,000/yr from age 65 onwards),
`annuity_balance` (drains by $12k/yr until exhausted), and
`annuity_payment` (constant nominal amount).

> **Note:** since `tax_kind` defaults to `"qualified"`, you don't
> need to touch the advanced group for this scenario.

### Modeling a non-qualified contract

Goal: same $200k contract, but funded with $50k of after-tax
money, so the first ~$2,500/yr of each payment is tax-free.

1. Set the four Simple-tier fields as above (`balance_today=200000`,
   `monthly_at_start=1000`, `start_age=65`, `lump_sum_mode=none`).
2. Toggle **"Show advanced"** in the sidebar to expose the
   advanced groups.
3. In **"Annuity (advanced)"**:
   - **Annuity tax kind** → `"Non-qualified (after-tax basis;
     §72 exclusion ratio)"`.
   - **Annuity cost basis $** → `50000`.
   - **Expected payout years** → `20` (controls the exclusion-
     ratio denominator — see [Knob reference](#knob-reference)).
   - **Annuity growth rate** → leave blank to default to
     `cfg.inflation`, or enter `0.04` for a 4% fixed-rate contract.
4. Click **Run**.

In the **Year-by-year** table you'll see `annuity_tax_free`
≈ $2,500/yr and `annuity_taxable` ≈ $9,500/yr from age 65 to 84
(when basis is exhausted). After that, `annuity_taxable` jumps to
the full $12,000.

### Electing a pension lump-sum

Goal: roll your $300k pension cash balance into a pretax IRA at
NRD (age 65) instead of taking the monthly annuity.

1. Configure the pension as usual: **Pension balance today** =
   `300000`, **Pension start age (NRD)** = `65`.
2. Set **Pension lump-sum election** to
   `"Rollover to pretax IRA (tax-free)"`.
3. Click **Run**.

What you'll see in the year-by-year output for age 65:

- `pension_lump_sum` ≈ pension balance (after any pre-NRD growth).
- `pension_lump_sum_event` = `"rollover_pretax"`.
- `pension` (current-year monthly income) drops to $0 — and stays
  $0 for all future years.
- `pretax_a_balance` jumps by approximately the lump-sum amount.
- `early_distribution_penalty` = $0 (rollovers are exempt).

To take it as cash instead, change the selector to
`"Cash lump sum (ordinary tax + 10% if pre-59.5)"`. If your
configured `start_age` is below 60, you'll also see
`early_distribution_penalty` = 10% of the lump-sum amount in that
year's row.

---

## Configuring via scenario JSON

Scenario files (`scenarios/*.json`) are the durable, sharable way
to capture an annuity setup. Both the CLI and the Dash app can load
them.

### Minimal annuity block

The complete `inputs.annuity` block, with every field at its
default:

```json
{
  "inputs": {
    "annuity": {
      "balance_today": 0.0,
      "monthly_at_start": 0.0,
      "start_age": 65,
      "growth_rate": null,
      "tax_kind": "qualified",
      "cost_basis": 0.0,
      "expected_payout_years": 20,
      "lump_sum_mode": "none"
    }
  }
}
```

You can omit any field you don't want to override — the loader
reads `Inputs`'s dataclass surface and uses the dataclass default
for any missing key. The above is equivalent to omitting the entire
`annuity` block.

### Pension lump-sum election

Add `lump_sum_mode` to the existing `inputs.pension` block:

```json
{
  "inputs": {
    "pension": {
      "balance_today": 350000,
      "monthly_at_nrd": 1700,
      "start_age": 65,
      "lump_sum_mode": "rollover_pretax"
    }
  }
}
```

Existing scenarios (without `lump_sum_mode`) keep working
identically — `"none"` is the default.

### Two ready-to-run examples

Two new example scenarios ship with the package:

- **`scenarios/example_annuity_nonqualified.json`** — A 60-year-old
  MFJ couple with a $200k non-qualified annuity ($50k cost basis)
  paying $1,000/mo from 65, with 4% pre-payout growth and a
  20-year expected payout. Shows the §72(b) exclusion-ratio split
  end-to-end.

- **`scenarios/example_pension_lump_sum.json`** — A 60-year-old
  couple with a $350k pension that's elected as a pretax IRA
  rollover at NRD. Demonstrates the rollover firing at age 65,
  the pension column going to zero from then on, and downstream
  RMDs picking up the rolled balance at age 75.

Run either with:

```bash
tax-optimizer --scenario scenarios/example_annuity_nonqualified.json
tax-optimizer --scenario scenarios/example_pension_lump_sum.json
```

### Loading a custom scenario

Two ways to load a scenario file you've written:

```bash
# CLI: pass --scenario PATH
tax-optimizer --scenario scenarios/my_plan.json

# Dash app: use the "Load scenario" button in the sidebar
tax-optimizer-app
# → click "Load scenario..." and pick the file
```

The Dash app fully populates every form field from the JSON, so
you can tweak from there interactively. Use **"Save scenario..."**
to write the current form state back to JSON.

> **Tip:** copy `scenarios/template.json` as a starting point.
> It includes every Config and Inputs field with its default
> value, so you only need to change the knobs you care about and
> delete the rest (or leave them — JSON allows redundant fields).

---

## Configuring from the CLI / Python API

### `--set` overrides

For one-off experiments without editing a JSON file, the CLI
accepts dotted-path overrides via `--set`:

```bash
# Add a $200k qualified annuity to the example01 scenario
tax-optimizer --scenario scenarios/example01.json \
    --set inputs.annuity.balance_today=200000 \
    --set inputs.annuity.monthly_at_start=1000 \
    --set inputs.annuity.start_age=65

# Take the pension as a cash lump sum
tax-optimizer --scenario scenarios/example01.json \
    --set inputs.pension.lump_sum_mode=cash

# Non-qualified annuity surrender
tax-optimizer --scenario scenarios/example01.json \
    --set inputs.annuity.balance_today=200000 \
    --set inputs.annuity.cost_basis=50000 \
    --set inputs.annuity.tax_kind=non_qualified \
    --set inputs.annuity.lump_sum_mode=cash
```

Each `--set` argument is a single `dotted.path=value` pair.
Strings are accepted bare (no quotes needed); numbers and booleans
are auto-coerced.

### Python: building `Inputs` programmatically

If you're driving the simulator from a notebook or a script:

```python
from tax_optimizer import Config, Inputs
from tax_optimizer.inputs import AnnuityInputs, PensionInputs
from tax_optimizer.simulator import simulate

# A non-qualified annuity, with monthly payments
inp = Inputs(
    annuity=AnnuityInputs(
        balance_today=200_000,
        monthly_at_start=1_000,
        start_age=65,
        tax_kind="non_qualified",
        cost_basis=50_000,
        expected_payout_years=20,
        growth_rate=0.04,
    ),
)

# A pension with a rollover election
inp_pension_rollover = Inputs(
    pension=PensionInputs(
        balance_today=350_000,
        monthly_at_nrd=1_700,
        start_age=65,
        lump_sum_mode="rollover_pretax",
    ),
)

df = simulate(Config(), inp)
print(df[
    ["spouse_a_age", "annuity_balance", "annuity_taxable",
     "annuity_tax_free", "annuity_basis_remaining"]
].head(25))
```

`Inputs.__post_init__` validates the configuration at construction
time, so a bad `tax_kind` or the forbidden `non_qualified +
rollover_pretax` combination raises immediately rather than
producing wrong tax math at simulate time.

---

## Knob reference

### `inputs.pension.lump_sum_mode`

| Value | Behavior |
|---|---|
| `"none"` *(default)* | Existing monthly-pension behavior. |
| `"rollover_pretax"` | At `start_age`, balance moves to pretax IRA. No current-year tax. |
| `"cash"` | At `start_age`, balance distributed as ordinary income. 10% surtax under 59½. |

### `inputs.annuity` block

| Field | Type | Default | Meaning |
|---|---|---|---|
| `balance_today` | float | `0.0` | Current contract value in today's dollars. |
| `monthly_at_start` | float | `0.0` | Fixed monthly payment after `start_age` (no COLA, matching pension convention). |
| `start_age` | int | `65` | Age at which payments begin (or, with a lump-sum election, the year the contract is liquidated). |
| `growth_rate` | float \| null | `null` | Pre-payout balance growth rate. `null` defers to `cfg.inflation` (zero real growth). Set to e.g. `0.04` for a 4% fixed-rate contract. |
| `tax_kind` | `"qualified"` \| `"non_qualified"` | `"qualified"` | See below. |
| `cost_basis` | float | `0.0` | After-tax investment in the contract (non-qualified only; ignored for qualified). |
| `expected_payout_years` | int | `20` | §72(b) exclusion-ratio denominator. Shorter = faster basis recovery; longer = more tax-free dollars per payment for more years. |
| `lump_sum_mode` | `"none"` \| `"rollover_pretax"` \| `"cash"` | `"none"` | Same semantics as the pension field. |

#### `tax_kind` deep-dive

- **`"qualified"`**: every distribution dollar is ordinary income.
  Models an annuity inside a 401(k), 403(b), or IRA. `cost_basis`
  and `expected_payout_years` are ignored.
- **`"non_qualified"`**: tracks a cost basis and applies the IRC
  §72(b) exclusion ratio:

    `tax_free_per_year = annual_payment × (cost_basis / (annual_payment × expected_payout_years))`
    `taxable_per_year  = annual_payment − tax_free_per_year`

  Once cumulative tax-free amounts equal `cost_basis`, every
  subsequent payment becomes 100% taxable.

#### `expected_payout_years` and the simplification

Real §72 uses IRS Treas. Reg. §1.72-9 life-expectancy tables. We
use a user-provided `expected_payout_years` instead — easier to
reason about and tweak. Reasonable values:

- **20** (default) — generic immediate annuity at typical
  retirement ages. Gives the example numbers you see in this
  guide.
- **10–15** — period-certain contracts with a fixed payment
  window.
- **25–30** — joint-life or younger-age contracts.

---

## Validation: errors you might hit

`Inputs(...)` raises immediately on bad configuration, so you
catch errors before the simulator runs:

| Error message (excerpt) | Cause | Fix |
|---|---|---|
| `"Non-qualified annuity cannot use lump_sum_mode='rollover_pretax'..."` | IRC prohibits this combo. | Use `"cash"` (taxable surrender) for non-qualified contracts. |
| `"inputs.annuity.tax_kind must be 'qualified' or 'non_qualified'..."` | Typo in `tax_kind`. | Use one of the two literal strings. |
| `"inputs.annuity.lump_sum_mode must be one of ..."` | Typo in `lump_sum_mode`. | `"none"` / `"rollover_pretax"` / `"cash"`. |
| `"inputs.pension.lump_sum_mode must be one of ..."` | Same as above, on the pension side. | Same fix. |
| `"inputs.annuity.expected_payout_years must be > 0..."` | Set to 0 or negative. | Use a positive integer (default 20). |
| `"inputs.annuity.cost_basis must be >= 0..."` | Negative cost basis. | Use 0 or a positive number. |

---

## Reading the output

The simulator's per-year DataFrame gains these columns when an
annuity or lump-sum election is active. They're zero in the
default (no-annuity, no-lump-sum) case, so existing reports stay
clean.

| Column | What it means |
|---|---|
| `annuity_balance` | End-of-year contract balance. Drains as payments are taken; jumps to 0 on a lump-sum event. |
| `annuity_basis_remaining` | Un-recovered §72(b) cost basis. Stays at 0 for qualified contracts. Drains on each non-qualified payment. |
| `annuity_payment` | Stored annual nominal payment (matches `monthly_at_start × 12`). |
| `annuity_taxable` | Ordinary-taxable portion of this year's annuity income. |
| `annuity_tax_free` | Tax-free basis-return portion (non-qualified only). |
| `annuity_lump_sum` | Lump-sum amount this year (only on the firing year; 0 otherwise). |
| `annuity_lump_sum_event` | `""`, `"rollover_pretax"`, or `"cash"`. |
| `pension_lump_sum`, `pension_lump_sum_event` | Same as above, for the pension. |
| `early_distribution_penalty` | Combined 10% §72(t)/§72(q) surtax dollars for the year. |

In the Dash **Year-by-year** tab you can sort and filter on any of
these columns; in the **Overview** and **Taxes** tabs the existing
KPI tiles already roll annuity income into the relevant aggregates
(taxes paid, marginal rate, AGI, etc.).

For state tax: annuity taxable income is treated identically to
pension income, so retirement-income exclusions (NY $20k/filer at
59½+, IL full exclusion, etc.) apply automatically.

---

## FAQ / common questions

**Q: I'm under 59½ and want to take my pension as cash. What's the
total tax bill?**

A: Federal ordinary income tax on the full balance at your
marginal rate, plus 10% IRC §72(t) additional tax on the same
amount, plus state ordinary income tax (subject to whatever
retirement-income exclusion your state offers). The simulator
shows each piece separately: `federal_tax`, `state_tax`, and
`early_distribution_penalty` (the 10% surtax) all appear as
separate columns in the year-by-year output.

**Q: What about §72(t) "substantially equal periodic payments"
exceptions?**

A: Not modeled. The simulator treats every pre-60 cash lump sum
as subject to the 10% surtax. If you're planning a 72(t) SEPP,
either (a) configure the contract for monthly payments instead
of a lump sum (no surtax applies to scheduled payments), or
(b) zero out the `early_distribution_penalty` column manually in
post-processing. A future release may add an explicit
"§72(t) SEPP exemption" flag if there's user demand.

**Q: My non-qualified annuity has a higher `cost_basis` than
`balance_today` (I lost money on the contract). What happens?**

A: The simulator clamps the exclusion ratio at 1.0 (every payment
fully tax-free) and drains the basis as usual. When the contract
is exhausted, payments stop. The remaining un-recovered basis is
not modeled as a deductible loss — that's an IRS Schedule A
miscellaneous deduction in the real world, but it's out of scope
here.

**Q: Can I model a variable annuity or an indexed annuity?**

A: Not directly. The annuity here is a **fixed-rate** contract:
`growth_rate` is a single deterministic number, and
`monthly_at_start` is nominal-fixed (no COLA, no market-linked
adjustments). For a variable annuity, set `growth_rate` to your
expected long-run return and accept the deterministic
approximation, or model the contract as a regular taxable account
and ignore the §72(b) exclusion mechanics.

**Q: What if I have multiple annuity contracts?**

A: The model supports a single contract via `inputs.annuity`. For
multiple contracts, the simplest approach is to aggregate them:
sum the balances, average the start ages, and pick a representative
`tax_kind`. The IRS aggregation rule for multiple non-qualified
contracts (which prevents using a separate exclusion ratio per
contract) isn't modeled.

**Q: Does the lump-sum cash distribution interact with Roth
conversions or RMDs?**

A: Yes — automatically. A cash lump sum increases the year's AGI,
which:

- May push you into a higher Roth conversion bracket (if
  `withdrawal_strategy="bracket_fill"`).
- Counts toward IRMAA for two years later.
- Doesn't affect the same-year RMD (RMDs are computed off prior-
  year-end pretax balances).

A rollover, by contrast, increases your pretax balance, which:

- **Increases all future RMDs** for that spouse.
- Doesn't affect any current-year tax line.

The Strategies tab in the Dash app is a good place to compare
side-by-side: configure two scenarios (cash vs. rollover) and look
at the terminal NW and total lifetime tax columns.
