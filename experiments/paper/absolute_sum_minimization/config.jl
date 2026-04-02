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
"""

# ============================================================================
# SMALL EXPERIMENTS (fast, for testing)
# ============================================================================
small_experiments = [
    Dict(
        "name" => "p2_1var_2poly_lin",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 3,
    ),
    Dict(
        "name" => "p2_1var_2poly_quad",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 2,
        "num_samples" => 3,
    ),
    Dict(
        "name" => "p2_2var_3poly_lin",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 3,
    ),
    Dict(
        "name" => "p2_5var_5poly_lin",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 3,
    ),
]

# ============================================================================
# COMPREHENSIVE SWEEP (vary all dimensions)
# ============================================================================
comprehensive_experiments = [
    # 2 polynomials, varying variables and degrees
    Dict("name" => "p2_1var_2poly_lin", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_1var_2poly_quad", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p2_2var_2poly_lin", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 2, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_2var_2poly_quad", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 2, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p2_3var_2poly_lin", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 3, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_3var_2poly_quad", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 3, "degree" => 2, "num_samples" => 5),

    # 3 polynomials, varying variables and degrees
    Dict("name" => "p2_1var_3poly_lin", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 1, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_1var_3poly_quad", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 1, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p2_2var_3poly_lin", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 2, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_2var_3poly_quad", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 2, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p2_3var_3poly_lin", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 3, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_3var_3poly_quad", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 3, "degree" => 2, "num_samples" => 5),
]

# ============================================================================
# PRIME COMPARISON (same problem, different primes)
# ============================================================================
prime_comparison = [
    Dict("name" => "p2_1var_2poly_lin", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p3_1var_2poly_lin", "prime" => 3, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p5_1var_2poly_lin", "prime" => 5, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5),
]

# ============================================================================
# SELECT WHICH SET TO USE
# ============================================================================

# Default: use small experiments for quick testing
experiment_configs = small_experiments

# Uncomment one of these to use a different experiment set:
# experiment_configs = comprehensive_experiments
# experiment_configs = prime_comparison

# Or define your own custom experiments:
# experiment_configs = [
#     Dict("name" => "custom", "prime" => 2, "prec" => 20,
#          "num_polys" => 4, "num_vars" => 2, "degree" => 2, "num_samples" => 10),
# ]
