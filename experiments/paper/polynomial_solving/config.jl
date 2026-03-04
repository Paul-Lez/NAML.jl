"""
Polynomial Solving Experiment Configuration

Minimize |f(z)| where f is a polynomial with a guaranteed root in Z_p^n.

Each config should have:
- name: Descriptive name for the experiment
- prime: The prime p for the p-adic field
- prec: The p-adic precision
- num_vars: Number of variables (1, 2, or 3)
- degree: Degree of the polynomial (1, 2, or 3)
- num_samples: Number of random instances to run
- opt_degree: Degree parameter for optimization (used in children())
"""

# ============================================================================
# SMALL EXPERIMENTS (fast, for testing)
# ============================================================================
small_experiments = [
    # 1 variable
    Dict("name" => "1var_deg1_2adic", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 1, "num_samples" => 3, "opt_degree" => 1),
    Dict("name" => "1var_deg2_2adic", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 2, "num_samples" => 3, "opt_degree" => 1),
    Dict("name" => "1var_deg3_2adic", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 3, "num_samples" => 3, "opt_degree" => 1),

    # 2 variables
    Dict("name" => "2var_deg1_2adic", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 1, "num_samples" => 3, "opt_degree" => 1),
    Dict("name" => "2var_deg2_2adic", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 2, "num_samples" => 3, "opt_degree" => 1),

    # 3 variables
    Dict("name" => "3var_deg1_2adic", "prime" => 2, "prec" => 20,
         "num_vars" => 3, "degree" => 1, "num_samples" => 3, "opt_degree" => 1),
]

# ============================================================================
# COMPREHENSIVE SWEEP (all degree x variable combinations)
# ============================================================================
comprehensive_experiments = [
    # 1 variable, degrees 1-3
    Dict("name" => "1var_deg1", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "1var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "1var_deg3", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 3, "num_samples" => 5, "opt_degree" => 1),

    # 2 variables, degrees 1-3
    Dict("name" => "2var_deg1", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "2var_deg3", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 3, "num_samples" => 5, "opt_degree" => 1),

    # 3 variables, degrees 1-3
    Dict("name" => "3var_deg1", "prime" => 2, "prec" => 20,
         "num_vars" => 3, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "3var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 3, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "3var_deg3", "prime" => 2, "prec" => 20,
         "num_vars" => 3, "degree" => 3, "num_samples" => 5, "opt_degree" => 1),
]

# ============================================================================
# PRIME COMPARISON
# ============================================================================
prime_comparison = [
    Dict("name" => "1var_deg2_2adic", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "1var_deg2_3adic", "prime" => 3, "prec" => 15,
         "num_vars" => 1, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
    Dict("name" => "1var_deg2_5adic", "prime" => 5, "prec" => 12,
         "num_vars" => 1, "degree" => 2, "num_samples" => 5, "opt_degree" => 1),
]

# ============================================================================
# SELECT WHICH SET TO USE
# ============================================================================

# Default: use small experiments for quick testing
experiment_configs = small_experiments
