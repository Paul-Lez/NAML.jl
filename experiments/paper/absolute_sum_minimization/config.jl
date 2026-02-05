"""
Absolute Sum Minimization Experiment Configuration

Define experiment configurations for minimizing sums of absolute polynomials.
For each configuration, we minimize: |f₁(x)| + |f₂(x)| + ... + |fₙ(x)|
where f₁, ..., fₙ are polynomials with random fixed coefficients.

Each config should have:
- name: Descriptive name for the experiment
- prime: The prime p for the p-adic field
- prec: The p-adic precision
- num_polys: Number of polynomials in the sum (e.g., 2 or 3)
- num_vars: Number of variables (dimension of x) (e.g., 1, 2, or 3)
- degree: Degree of each polynomial (1=linear, 2=quadratic)
- num_samples: Number of random instances to run
- opt_degree: Degree parameter for optimization (used in children())
"""

# ============================================================================
# SMALL EXPERIMENTS (fast, for testing)
# ============================================================================
small_experiments = [
    Dict(
        "name" => "2poly_1var_linear_2adic",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 3,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "2poly_1var_quadratic_2adic",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 2,
        "num_samples" => 3,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "3poly_2var_linear_2adic",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 3,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "5poly_5var_linear_2adic",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 3,
        "opt_degree" => 2
    ),
]

# ============================================================================
# COMPREHENSIVE SWEEP (vary all dimensions)
# ============================================================================
comprehensive_experiments = [
    # 2 polynomials, varying variables and degrees
    Dict("name" => "2poly_1var_linear", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_1var_quadratic", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_2var_linear", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 2, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_2var_quadratic", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 2, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_3var_linear", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 3, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_3var_quadratic", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 3, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),

    # 3 polynomials, varying variables and degrees
    Dict("name" => "3poly_1var_linear", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 1, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "3poly_1var_quadratic", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 1, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "3poly_2var_linear", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 2, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "3poly_2var_quadratic", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 2, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "3poly_3var_linear", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 3, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "3poly_3var_quadratic", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 3, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
]

# ============================================================================
# PRIME COMPARISON (same problem, different primes)
# ============================================================================
prime_comparison = [
    Dict("name" => "2poly_1var_linear_2adic", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_1var_linear_3adic", "prime" => 3, "prec" => 15,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_1var_linear_5adic", "prime" => 5, "prec" => 12,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
]

# ============================================================================
# DEGREE PARAMETER SWEEP (test different optimization degrees)
# ============================================================================
opt_degree_sweep = [
    Dict("name" => "2poly_2var_linear_deg1", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 2, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2poly_2var_linear_deg2", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 2, "degree" => 1, "num_samples" => 5, "opt_degree" => 2),
    Dict("name" => "2poly_2var_linear_deg3", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 2, "degree" => 1, "num_samples" => 5, "opt_degree" => 3),
]

# ============================================================================
# SELECT WHICH SET TO USE
# ============================================================================

# Default: use small experiments for quick testing
experiment_configs = small_experiments

# Uncomment one of these to use a different experiment set:
# experiment_configs = comprehensive_experiments
# experiment_configs = prime_comparison
# experiment_configs = opt_degree_sweep

# Or define your own custom experiments:
# experiment_configs = [
#     Dict("name" => "custom", "prime" => 2, "prec" => 20,
#          "num_polys" => 4, "num_vars" => 2, "degree" => 2, "num_samples" => 10, "opt_degree" => 1),
# ]
