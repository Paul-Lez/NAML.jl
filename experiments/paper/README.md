# Paper Experiments - Benchmarks and Utilities

This directory contains benchmark infrastructure, experiment utilities, and organized subdirectories for paper-ready experiments using the NAML library.

## Directory Structure

```
experiments/paper/
├── util.jl                          # Shared utility functions
├── test_util.jl                     # Tests for utilities
├── generate_paper_tables.sh         # Run all experiments and regenerate LaTeX tables
├── run_and_deploy_tables.sh         # Run experiments, generate tables, copy to arXiv draft
├── polynomial_learning/             # Polynomial interpolation experiments
├── absolute_sum_minimization/       # Absolute sum optimization experiments
├── function_learning/               # Binary classification experiments
├── polynomial_solving/              # Root-finding / polynomial solving experiments
└── worked_examples/                 # Self-contained illustrative scripts
```

## Running Experiments

### Full Pipeline (Recommended)

To run all four experiment suites, regenerate all LaTeX tables, and copy them to the arXiv draft directory:

```bash
# From repository root
bash experiments/paper/run_and_deploy_tables.sh
```

To only run experiments and regenerate tables (without copying to the paper):

```bash
bash experiments/paper/generate_paper_tables.sh
```

Both scripts support the following flags:

| Flag | Description |
|------|-------------|
| `--quick` | Reduced epochs (5) and simulations (10/20/40) for a fast smoke-test |
| `--epochs N` | Override number of optimization epochs (default: 20) |
| `--move-only` | (run_and_deploy only) Skip experiments; just copy existing `.tex` files to the paper |

**Examples:**
```bash
# Smoke test (fast)
bash experiments/paper/run_and_deploy_tables.sh --quick

# Full run, override epochs
bash experiments/paper/generate_paper_tables.sh --epochs 50

# Just re-deploy already-generated tables
bash experiments/paper/run_and_deploy_tables.sh --move-only
```

### Per-Experiment Scripts

Each experiment subdirectory has a consistent interface:

```bash
# Standard flags (forwarded by the shell scripts above)
julia --project=. experiments/paper/<name>/run_experiments.jl [--quick] [--paper] [--save] [--epochs N] [--output FILE]

# Generate LaTeX tables from saved JSON results
julia --project=. experiments/paper/<name>/generate_tables.jl <results.json> [--output tables.tex]

# Quick sanity check (verifies pipeline, not paper quality)
julia --project=. experiments/paper/<name>/sanity_run.jl
```

`--paper` loads `paper_config.jl` (comprehensive, multi-sample configs).
`--config` loads `config.jl` (standard configs, faster than paper).
Without a config flag, a minimal default set of experiments runs.

### Paper Results Files

Each experiment directory contains pre-generated paper results (committed to the repo):

| Experiment | Results JSON | LaTeX tables |
|---|---|---|
| `absolute_sum_minimization/` | `absolute_sum_results_paper.json` | `absolute_sum_tables.tex` |
| `function_learning/` | `function_learning_results_paper.json` | `function_learning_tables.tex` |
| `polynomial_learning/` | `poly_learning_results_paper.json` | `polynomial_learning_tables.tex` |
| `polynomial_solving/` | `polynomial_solving_results_paper.json` | `polynomial_solving_tables.tex` |

---

## Core Utilities

### `util.jl` - Shared Experiment API

Provides reusable functions for all experiments:

- **`generate_random_padic(p, prec, min_exp, num_terms)`**: Generate random p-adic numbers with configurable exponent range
- **`polynomial_to_linear_loss(data, degree, cutoff_val)`**: Transform polynomial learning into linear optimization
  - Handles both p-adic outputs (`cutoff_val=nothing`) and real outputs (with cutoff)
- **`polynomial_to_crossentropy_loss(data, degree, threshold, scale)`**: Smooth binary classification loss
- **`polynomial_to_valuation_crossentropy_loss(data, degree, prime, threshold, scale)`**: Classification with p-adic valuation
- **`polynomial_to_mse_loss(data, degree)`**: Direct MSE regression
- **`generate_gauss_point(n, K)`**: Create Gauss points (standard starting point)
- **`generate_polynomial_learning_data(p, prec, n_points)`**: Generate training data with guaranteed distinct x values
- **`compute_classification_accuracy(model, data, param, threshold, scale)`**: Compute classification accuracy

### `test_util.jl` - Test Suite

Comprehensive tests for all utility functions:
- Random p-adic generation (including negative exponents)
- Uniqueness of generated data points
- Loss function creation (p-adic and real outputs)
- End-to-end optimization examples

---

## Experiment Subdirectories

### `polynomial_learning/` - Polynomial Interpolation

Learn polynomial coefficients from data. Varies degree and base prime.

**Files:**
- `run_experiments.jl` - Main launcher with JSON output
- `config.jl` / `paper_config.jl` / `sanity_config.jl` - Configurations
- `generate_tables.jl` - LaTeX table generator
- `sanity_run.jl` - MCTS vs DAG-MCTS comparison sanity check
- `polynomial_learning.ipynb` - Interactive Jupyter notebook

**Optimizers:** Greedy (deg 1 & 2), MCTS-50/100, DAG-MCTS-100, UCT, HOO

### `absolute_sum_minimization/` - Absolute Sum Minimization

Minimize sums of absolute polynomials: |f₁(x)| + |f₂(x)| + ... + |fₙ(x)|

**Files:**
- `run_experiments.jl` - Main launcher
- `config.jl` / `paper_config.jl` / `sanity_config.jl` - Configurations
- `generate_tables.jl` - LaTeX table generator
- `sanity_run.jl` - Quick sanity check
- `util.jl` - Problem-specific utilities
- `test_setup.jl` - Setup testing

**Optimizers:** Random, Greedy, MCTS-50/100/200

### `function_learning/` - Binary Classification

Learn a function that classifies p-adic inputs via cross-entropy loss.

**Files:**
- `run_experiments.jl` - Main launcher
- `config.jl` / `paper_config.jl` / `sanity_config.jl` - Configurations
- `generate_tables.jl` - LaTeX table generator
- `sanity_run.jl` - Quick sanity check
- `learn_zero_function.jl` - Learn constant 0 function
- `learn_one_function.jl` - Learn constant 1 function (uses DAG-MCTS)
- `function_learning.ipynb`, `function_learning_experiment.ipynb` - Notebooks

**Features:** Cross-entropy loss with sigmoid, classification accuracy computation

### `polynomial_solving/` - Root Finding

Minimize |f(z)| where f is a random polynomial with a guaranteed root in the Gauss point (unit ball). Tests varying numbers of variables (1–3) and polynomial degrees (1–3).

**Files:**
- `run_experiments.jl` - Main launcher
- `util.jl` - Polynomial generation with guaranteed roots
- `config.jl` / `paper_config.jl` / `sanity_config.jl` - Configurations
- `generate_tables.jl` - LaTeX table generator
- `sanity_run.jl` - Quick sanity check over all (vars, degree) combinations

**Optimizers:** Random, Best-First (deg 1 & 2), MCTS-50/100/200, DAG-MCTS-50/100/200, DOO, Gradient-Descent

**Paper config:** 27 experiments (3 primes × 3 variables × 3 degrees), 5 samples each.

### `worked_examples/` - Illustrative Scripts

Self-contained scripts demonstrating specific problem instances, intended for the paper's worked examples section.

**Files:**
- `cubic_sum_minimization.jl` - Minimize |(1-a)(1-2a)(1-4a)|₂ + ... over Q₂ (unique minimizer at a=1)
- `x2_minus_1_minimization.jl` - Minimize |x²-1| over Q₂

---

## Configuration System

Each experiment subdirectory uses three config levels:

| File | Purpose |
|------|---------|
| `sanity_config.jl` | Minimal configs for `sanity_run.jl` (smoke test) |
| `config.jl` | Standard configs (used with `--config` flag) |
| `paper_config.jl` | Comprehensive paper-ready configs (used with `--paper` flag) |

Configs are plain Julia files that define an `experiment_configs` vector of `Dict`s, each specifying `prime`, `prec`, `num_vars`/`degree`/`n_points`, `num_samples`, and `opt_degree`.

---

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "metadata": {
    "timestamp": "...",
    "n_epochs": 20,
    "quick_mode": false,
    "optimizer_order": [...]
  },
  "experiments": [
    {
      "config": { "name": "...", "prime": 2, ... },
      "samples": [ { "sample_num": 1, "initial_loss": ..., "optimizers": { ... } }, ... ],
      "aggregate": {
        "MCTS-100": {
          "mean_final_loss": ..., "std_final_loss": ...,
          "mean_improvement": ..., "mean_improvement_ratio": ...,
          "mean_time": ..., "min_final_loss": ..., "max_final_loss": ...
        }
      }
    }
  ]
}
```

`generate_tables.jl` in each directory reads this JSON and outputs LaTeX `tabular` environments ready to `\input{}` into the paper.
