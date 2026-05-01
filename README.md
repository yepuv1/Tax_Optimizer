# Retirement Tax Optimizer

A standalone Jupyter notebook that jointly optimizes retirement-tax decisions for a married-filing-jointly couple:

1. **Pre-retirement** allocation between Traditional 401(k) and Roth 401(k) for each spouse.
2. **In-retirement** withdrawal sequencing across taxable / pretax / Roth buckets.
3. **Roth-conversion** sizing during the retirement-to-RMD gap years.

The objective is to maximize after-tax terminal net worth at the planning horizon, subject to never running out of money, while modeling federal brackets, LTCG, NIIT, IRMAA, Social Security provisional-income taxation, and per-spouse RMD rules.

All inputs are hardcoded in cell §2 (`CFG = Config(...)`) — edit them to model your own scenario, then re-run the notebook from there.

## Requirements

- Python **3.10+**
- macOS, Linux, or Windows

## Installing Python 3

Check whether you already have Python 3.10+:

```bash
python3 --version
```

If the command is missing or reports an older version, install it:

### macOS

Use the official installer from [python.org/downloads](https://www.python.org/downloads/), or via Homebrew:

```bash
brew install python@3.12
```

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install -y python3 python3-pip
```

### Windows

Download the installer from [python.org/downloads](https://www.python.org/downloads/) and **check "Add python.exe to PATH"** during install. Verify with:

```powershell
py -3 --version
```

## Setup

Choose **one** of the two workflows below.

### Option A — Standard `venv` + `pip`

```bash
git clone https://github.com/vijayyepuri/Tax_Optimizer.git
cd Tax_Optimizer

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install -e ".[notebook]"
```

If you only want the simulation libraries (no Jupyter), `pip install -e .` is enough.

### Option B — `uv` (faster, optional)

[`uv`](https://docs.astral.sh/uv/) is an extremely fast Python package and project manager from Astral. It can install Python for you, create the virtual environment, and resolve dependencies in seconds.

**Install `uv`:**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via Homebrew
brew install uv
```

**Set up the project:**

```bash
git clone https://github.com/vijayyepuri/Tax_Optimizer.git
cd Tax_Optimizer

uv venv --python 3.12          # creates .venv (uv will download Python if needed)
source .venv/bin/activate       # Windows: .venv\Scripts\activate

uv pip install -e ".[notebook]"
```

You can also skip activating the venv and prefix commands with `uv run`, e.g.:

```bash
uv run jupyter lab tax_optimizer_standalone.ipynb
```

## Running the notebook

### Option 1 — JupyterLab (recommended)

```bash
jupyter lab tax_optimizer_standalone.ipynb
```

Then **Run → Run All Cells**.

### Option 2 — Classic Jupyter Notebook

```bash
jupyter notebook tax_optimizer_standalone.ipynb
```

### Option 3 — VS Code / Cursor

Open `tax_optimizer_standalone.ipynb` directly in the editor. When prompted, select the `.venv` interpreter as the kernel, then click **Run All**.

### Option 4 — Headless execution (no UI)

Execute the notebook end-to-end and write the results back in place:

```bash
jupyter nbconvert --to notebook --execute tax_optimizer_standalone.ipynb --inplace
```

Or render a static HTML report:

```bash
jupyter nbconvert --to html --execute tax_optimizer_standalone.ipynb
```

## Customizing your scenario

1. Open the notebook.
2. Edit cell §2 (`CFG = Config(...)`) — for example:
   - `spouse_a_retire_age=62`
   - `horizon_age=90`
   - pension, Social Security, and starting-balance fields
3. Re-run the notebook from cell §2 onward. The summary cells at the bottom rebuild themselves from the latest `results` dict, so the final write-up always reflects your inputs.

## What the notebook produces

- Side-by-side simulation of multiple strategies (e.g. all-Traditional, all-Roth, gap-year conversions, and a solver-optimized hybrid).
- Year-by-year cash-flow / balance / tax tables.
- Matplotlib charts comparing trajectories.
- A plain-English takeaways section that's regenerated from your inputs each run.

## Project layout

```
.
├── tax_optimizer_standalone.ipynb   # the notebook (all logic + inputs)
├── pyproject.toml                    # dependencies and project metadata
├── LICENSE
└── README.md
```

## Disclaimer

This notebook is for **educational and illustrative purposes only**. It is not tax, legal, or investment advice. Tax law changes frequently and individual situations vary — consult a qualified professional before acting on any output.

## License

See [LICENSE](LICENSE).
