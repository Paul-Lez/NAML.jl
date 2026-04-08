# Paper Experiments

> **Architecture**: Three-stage pipeline — **Run → Stats → Tables** — with shared utilities and threading.

## Quick Start

```bash
# Full pipeline: run experiments, compute stats, generate tables
bash experiments/paper/generate_paper_tables.sh

# Quick smoke test
bash experiments/paper/generate_paper_tables.sh --quick

# Full pipeline + copy to arXiv draft
bash experiments/paper/run_and_deploy_tables.sh

# Single experiment (e.g., absolute_sum_minimization)
julia --project=. experiments/paper/absolute_sum_minimization/run_experiments.jl --quick --save
julia --project=. experiments/paper/make_stats.jl experiments/paper/absolute_sum_minimization/*_raw.json
julia --project=. experiments/paper/absolute_sum_minimization/generate_tables.jl experiments/paper/absolute_sum_minimization/*_stats.json --stdout
```

## Architecture

### Three-Stage Pipeline

```
run_experiments.jl  →  *_raw.json
                            ↓
make_stats.jl       →  *_stats.json
                            ↓
generate_tables.jl  →  *.tex
```

1. **`run_experiments.jl`** — Runs experiments and logs raw per-sample results to JSON. No statistical aggregation. Optimizer execution is threaded (`Threads.@threads`) for parallel speedup.

2. **`make_stats.jl`** — Reads raw JSON, computes per-sample rankings, per-experiment aggregate statistics (mean/std/min/max), and cross-experiment global ranking. Writes stats JSON.

3. **`generate_tables.jl`** — Reads stats JSON and generates LaTeX tables. Experiment-specific tables (e.g., accuracy for function_learning) are defined locally; common tables are shared.

### Shared Utilities

All shared code lives in `experiments/paper/`:

| File | Purpose |
|------|---------|
| `experiment_utils.jl` | CLI parsing, optimizer factory, threaded runner, JSON save |
| `stats_utils.jl` | Mean/std, ranking, aggregate statistics |
| `table_utils.jl` | LaTeX formatting, display names, generic table generators |
| `util.jl` | Problem generation, loss functions, data utilities |

### File Structure

```
experiments/paper/
├── experiment_utils.jl       # Shared: CLI, optimizer factory, threading
├── stats_utils.jl            # Shared: statistics computation
├── table_utils.jl            # Shared: LaTeX table generation
├── make_stats.jl             # Global: raw JSON → stats JSON
├── util.jl                   # Shared: p-adic generation, loss, etc.
├── generate_paper_tables.sh  # Pipeline orchestrator
├── run_and_deploy_tables.sh  # Pipeline + copy to arXiv
├── README.md
│
├── absolute_sum_minimization/
│   ├── run_experiments.jl    # Experiment runner
│   ├── generate_tables.jl    # Table generator
│   ├── util.jl               # Local problem generation
│   ├── config.jl
│   ├── paper_config.jl
│   └── sanity_run.jl
│
├── function_learning/
│   ├── run_experiments.jl
│   ├── generate_tables.jl
│   ├── config.jl
│   ├── paper_config.jl
│   └── sanity_run.jl
│
├── polynomial_learning/
│   ├── run_experiments.jl
│   ├── generate_tables.jl
│   ├── config.jl
│   ├── paper_config.jl
│   └── sanity_run.jl
│
├── polynomial_solving/
│   ├── run_experiments.jl
│   ├── generate_tables.jl
│   ├── util.jl               # Local: polynomial w/ guaranteed roots
│   ├── config.jl
│   ├── paper_config.jl
│   └── sanity_run.jl
│
└── worked_examples/
    └── ...
```

## CLI Reference

### run_experiments.jl (all 4 experiment dirs)

```bash
julia --project=. experiments/paper/<experiment>/run_experiments.jl [FLAGS]

Flags:
  --quick              Reduced epochs (5) and simulations for smoke testing
  --save               Save results to JSON file
  --config             Use configurations from config.jl
  --paper              Use paper-ready configurations from paper_config.jl
  --epochs N           Override number of epochs (default: 20)
  --output FILE        Override output filename
  --samples N          Override number of samples per config
  --selection-mode M   MCTS selection mode: BestValue, VisitCount, or BestLoss
  --degree D           Override tree branching degree
  --description TEXT   Experiment description (stored in JSON metadata)
  --git-commit HASH    Git commit hash (stored in JSON metadata)
```

### make_stats.jl

```bash
julia --project=. experiments/paper/make_stats.jl <raw.json> [--output stats.json]
```

Automatically detects experiment type from JSON metadata for type-specific processing (e.g., accuracy fields for function_learning).

### generate_tables.jl (all 4 experiment dirs)

```bash
julia --project=. experiments/paper/<experiment>/generate_tables.jl <stats.json> [FLAGS]

Flags:
  --output FILE   Output .tex filename (default: <experiment>_tables.tex)
  --stdout        Print tables to stdout instead of file
  --verbose       Include per-configuration detailed tables
```

### generate_paper_tables.sh

```bash
bash experiments/paper/generate_paper_tables.sh [FLAGS]

Flags:
  --quick              Smoke test mode
  --epochs N           Override epochs
  --samples N          Override samples (default: 30)
  --selection-mode M   MCTS selection mode
  --degree D           Override tree degree
  --verbose            Include detailed tables
```

## Threading

Experiment runners use `Threads.@threads` to parallelize optimizer execution within each sample. To use multiple threads:

```bash
julia -t 4 --project=. experiments/paper/<experiment>/run_experiments.jl --quick --save
```

Each thread deep-copies the starting parameter and loss to avoid mutation issues.

## JSON Schema

### Raw JSON (`*_raw.json`)

```json
{
  "metadata": {
    "experiment_type": "absolute_sum_minimization",
    "timestamp": "2024-01-01 12:00:00",
    "n_epochs": 20,
    "quick_mode": false,
    "optimizer_order": ["Random", "Best-First", ...],
    "description": "",
    "git_commit": ""
  },
  "experiments": [
    {
      "config": { "name": "...", "prime": 2, ... },
      "samples": [
        {
          "sample_num": 1,
          "initial_loss": 1.23,
          "optimizers": {
            "Random": { "time": 0.5, "final_loss": 0.8, "losses": [...], "improvement": 0.43, "improvement_ratio": 0.35, "total_evals": 100 },
            ...
          }
        }
      ]
    }
  ]
}
```

### Stats JSON (`*_stats.json`)

Same as raw JSON, plus:
- Each sample's optimizers get a `"rank"` field
- Each experiment gets an `"aggregate"` dict (mean/std/min/max per optimizer)
- Top-level `"global_ranking"` dict (avg rank across configs)

## Experiment Types

| Experiment | Description | Extra Fields |
|------------|-------------|--------------|
| `absolute_sum_minimization` | Minimize `\|f₁(x)\| + \|f₂(x)\| + ...` | — |
| `function_learning` | Learn binary classifier via cross-entropy | `final_accuracy`, `accuracy_improvement` |
| `polynomial_learning` | Learn polynomial coefficients from `(x, y)` data | — |
| `polynomial_solving` | Minimize `\|f(z)\|` where `f` has a known root | — |
