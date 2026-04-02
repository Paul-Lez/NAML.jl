"""
Function Learning Experiment Configuration

Define experiment configurations for learning target functions over p-adic fields.
For each configuration, we learn polynomial coefficients (a₀, ..., aₙ) such that
    f(x) = a₀ + a₁x + ... + aₙxⁿ ≈ target_function(x)
for random p-adic inputs x.

Three main tasks:
1. Zero Function: Learn f(x) = 0 for all x (trivial solution exists)
2. One Function: Learn f(x) = 1 for all x (requires constant polynomial)
3. Random Function: Learn to approximate random binary labels (0 or 1)

Each config should have:
- name: Descriptive name for the experiment
- prime: The prime p for the p-adic field
- prec: The p-adic precision
- degree: Polynomial degree to fit
- n_points: Number of random test points
- target_fn: Target function ("zero", "one", or "random")
- num_samples: Number of random problem instances to average over
- threshold: Threshold for cross-entropy loss (optional)
- scale: Scale parameter for cross-entropy loss (optional)
"""

# ============================================================================
# SMALL EXPERIMENTS (fast, for testing)
# ============================================================================
small_experiments = [
    Dict("name" => "p3_zero_deg3", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_one_deg3", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 3, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# ZERO FUNCTION EXPERIMENTS (varying degree)
# ============================================================================
zero_function_sweep = [
    Dict("name" => "p2_zero_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_zero_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_zero_deg4", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_zero_deg5", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 7, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# ONE FUNCTION EXPERIMENTS (varying degree)
# ============================================================================
one_function_sweep = [
    Dict("name" => "p2_one_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_one_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_one_deg4", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_one_deg5", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 7, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# PRIME SWEEP (same function, different primes)
# ============================================================================
prime_sweep = [
    Dict("name" => "p2_zero_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_zero_deg3", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p5_zero_deg3", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p7_zero_deg3", "prime" => 7, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# RANDOM FUNCTION EXPERIMENTS (varying degree)
# ============================================================================
random_function_sweep = [
    Dict("name" => "p2_random_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_random_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_random_deg4", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_random_deg5", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 7, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# COMPREHENSIVE (all three functions, multiple settings)
# ============================================================================
comprehensive = [
    # Zero function - 2-adic
    Dict("name" => "p2_zero_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_zero_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_zero_deg4", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 2-adic
    Dict("name" => "p2_one_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_one_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_one_deg4", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Random function - 2-adic
    Dict("name" => "p2_random_deg2", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_random_deg3", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p2_random_deg4", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Zero function - 3-adic
    Dict("name" => "p3_zero_deg2", "prime" => 3, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_zero_deg3", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_zero_deg4", "prime" => 3, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 3-adic
    Dict("name" => "p3_one_deg2", "prime" => 3, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_one_deg3", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_one_deg4", "prime" => 3, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Random function - 3-adic
    Dict("name" => "p3_random_deg2", "prime" => 3, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_random_deg3", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p3_random_deg4", "prime" => 3, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Zero function - 5-adic
    Dict("name" => "p5_zero_deg2", "prime" => 5, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p5_zero_deg3", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p5_zero_deg4", "prime" => 5, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 5-adic
    Dict("name" => "p5_one_deg2", "prime" => 5, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p5_one_deg3", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p5_one_deg4", "prime" => 5, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Random function - 5-adic
    Dict("name" => "p5_random_deg2", "prime" => 5, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p5_random_deg3", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "p5_random_deg4", "prime" => 5, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# ============================================================================
# SELECT WHICH SET TO USE
# ============================================================================

# Default: use small experiments for quick testing
# experiment_configs = small_experiments

# Uncomment one of these to use a different experiment set:
# experiment_configs = zero_function_sweep
# experiment_configs = one_function_sweep
# experiment_configs = random_function_sweep
# experiment_configs = prime_sweep
experiment_configs = comprehensive
