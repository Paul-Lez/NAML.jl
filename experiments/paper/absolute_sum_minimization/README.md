# Absolute Sum Minimization Experiments

Experiments for minimizing sums of absolute polynomials over p-adic fields.

## Problem Setup

For polynomials f₁, f₂, ..., fₙ with fixed random coefficients in a p-adic field,
find x that minimizes:

```
L(x) = |f₁(x)| + |f₂(x)| + ... + |fₙ(x)|
```

Key aspects:
- **Polynomial coefficients are fixed** (randomly generated)
- **Variable x is optimized** (starts at Gauss point)
- **No training data** - pure optimization problem
- Uses `LinearAbsolutePolynomialSum` for linear case (optimized performance)
- Uses `AbsolutePolynomialSum` for higher degrees

## Quick Start

```bash
# Run with default settings (quick test)
julia --project=. run_experiments.jl

# Run with quick mode (fewer epochs)
julia --project=. run_experiments.jl --quick

# Run with config file
julia --project=. run_experiments.jl --config

# Save results to JSON
julia --project=. run_experiments.jl --config --save

# Quick test with config and save
julia --project=. run_experiments.jl --config --quick --save
```

## Configuration

Edit `config.jl` to customize experiments. Each configuration specifies:

```julia
Dict(
    "name" => "experiment_name",       # Descriptive name
    "prime" => 2,                      # Prime p for p-adic field
    "prec" => 20,                      # p-adic precision
    "num_polys" => 2,                  # Number of polynomials in sum
    "num_vars" => 1,                   # Number of variables (dimension)
    "degree" => 1,                     # Polynomial degree (1=linear, 2=quadratic)
    "num_samples" => 5,                # Number of random instances
    "opt_degree" => 1                  # Degree parameter for optimization
)
```

### Pre-defined Configuration Sets

- `small_experiments` - Fast tests with 3 samples each
- `comprehensive_experiments` - Full sweep: 2-3 polys, 1-3 vars, linear-quadratic
- `prime_comparison` - Same problem across 2-adic, 3-adic, 5-adic fields
- `opt_degree_sweep` - Test different optimization degree parameters

To use a different set, edit the bottom of `config.jl`:

```julia
experiment_configs = comprehensive_experiments  # Change this line
```

## Command Line Flags

- `--quick` - Reduce epochs to 5 and MCTS simulations to 10 (fast testing)
- `--save` - Save results to timestamped JSON file
- `--config` - Load experiments from `config.jl` (otherwise uses defaults)

## Output

### Console Output

For each experiment and sample:
```
Experiment: 2poly_1var_linear
Prime: 2, Polynomials: 2, Variables: 1, Degree: 1
Samples: 3, Optimization degree: 1

  [Sample 1/3]
    Initial: 1.234567e+00
    Greedy     Final: 2.345678e-01 (Δ: 9.999999e-01, 81.0%)
    MCTS       Final: 1.234567e-01 (Δ: 1.111111e+00, 90.0%)
```

### Summary Table

After all experiments, shows aggregate statistics:
```
Experiment 1: 2poly_1var_linear
  Polynomials: 2, Variables: 1, Degree: 1, Samples: 5

  Optimizer        Mean Final    Mean Improv.      Improv. %    Time (s)
  ---------------------------------------------------------------------------
  Greedy          1.234567e-01    9.876543e-01          80.0%        0.12
  MCTS            5.678901e-02    1.123456e+00          95.0%        1.45
```

### JSON Output (--save flag)

Saved to `absolute_sum_results_YYYYMMDD_HHMMSS.json`:

```json
[
  {
    "config": {
      "name": "2poly_1var_linear",
      "prime": 2,
      "num_polys": 2,
      "num_vars": 1,
      "degree": 1,
      ...
    },
    "samples": [
      {
        "sample_num": 1,
        "initial_loss": 1.234,
        "optimizers": {
          "Greedy": {
            "time": 0.15,
            "final_loss": 0.123,
            "losses": [1.234, 0.789, 0.456, 0.123],
            "improvement": 1.111,
            "improvement_ratio": 0.90
          },
          "MCTS": { ... }
        }
      },
      ...
    ],
    "aggregate": {
      "Greedy": {
        "mean_final_loss": 0.123,
        "mean_improvement": 1.111,
        "mean_improvement_ratio": 0.90,
        "mean_time": 0.15,
        "std_final_loss": 0.01,
        "min_final_loss": 0.10,
        "max_final_loss": 0.15
      },
      "MCTS": { ... }
    }
  },
  ...
]
```

## Optimizers

Current optimizers:
- **Greedy Descent** - Tree-based greedy optimization
- **MCTS** - Monte Carlo Tree Search

Parameters can be adjusted in `get_optimizer_configs()` function in `run_experiments.jl`.

## Customization Examples

### Run specific problem size

```julia
# Edit config.jl or pass inline
experiment_configs = [
    Dict("name" => "large_problem", "prime" => 2, "prec" => 25,
         "num_polys" => 4, "num_vars" => 3, "degree" => 2,
         "num_samples" => 10, "opt_degree" => 2),
]
```

### Add more optimizers

Edit `get_optimizer_configs()` in `run_experiments.jl`:

```julia
function get_optimizer_configs(K, opt_degree)
    return Dict(
        "Greedy" => ...,
        "MCTS" => ...,
        "UCT" => Dict(
            "init" => (param, loss) -> begin
                config = NAML.UCTConfig(
                    max_depth=10,
                    num_simulations=50,
                    exploration_constant=1.41,
                    degree=opt_degree
                )
                NAML.uct_descent_init(param, loss, config)
            end
        ),
    )
end
```

### Adjust polynomial generation

Edit `util.jl` functions:
- `generate_random_linear_polynomial()` - Modify coefficient ranges
- `generate_random_polynomial()` - Add more monomials or higher degrees

## Files

- `run_experiments.jl` - Main experiment runner
- `config.jl` - Experiment configurations
- `util.jl` - Problem generation utilities
- `README.md` - This file

## Tips

1. Start with `--quick` flag to test setup
2. Use small experiments first to verify correctness
3. Increase `num_samples` for more statistical reliability
4. Higher `prec` gives more accuracy but slower computation
5. `opt_degree` affects optimization granularity (higher = finer search)
