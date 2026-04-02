"""
Paper-Ready Experiment Configuration for Polynomial Solving

Comprehensive experiments demonstrating:
1. Effect of polynomial degree (1, 2, 3) on solving difficulty
2. Effect of number of variables (1, 2, 3) on solving difficulty
3. Comparison across primes (2, 3, 5)
4. Comparison of optimizers
"""

# ============================================================================
# PAPER-READY EXPERIMENTS
# ============================================================================

paper_experiments = [
    # ========================================================================
    # GROUP 1: Prime = 2, full degree x variable grid
    # ========================================================================

    # 1 variable
    Dict("name" => "p2_1var_deg1", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_1var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p2_1var_deg3", "prime" => 2, "prec" => 20,
         "num_vars" => 1, "degree" => 3, "num_samples" => 5),

    # 2 variables
    Dict("name" => "p2_2var_deg1", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_2var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p2_2var_deg3", "prime" => 2, "prec" => 20,
         "num_vars" => 2, "degree" => 3, "num_samples" => 5),

    # 3 variables
    Dict("name" => "p2_3var_deg1", "prime" => 2, "prec" => 20,
         "num_vars" => 3, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p2_3var_deg2", "prime" => 2, "prec" => 20,
         "num_vars" => 3, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p2_3var_deg3", "prime" => 2, "prec" => 20,
         "num_vars" => 3, "degree" => 3, "num_samples" => 5),

    # ========================================================================
    # GROUP 2: Prime = 3, full degree x variable grid
    # ========================================================================

    # 1 variable
    Dict("name" => "p3_1var_deg1", "prime" => 3, "prec" => 20,
         "num_vars" => 1, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p3_1var_deg2", "prime" => 3, "prec" => 20,
         "num_vars" => 1, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p3_1var_deg3", "prime" => 3, "prec" => 20,
         "num_vars" => 1, "degree" => 3, "num_samples" => 5),

    # 2 variables
    Dict("name" => "p3_2var_deg1", "prime" => 3, "prec" => 20,
         "num_vars" => 2, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p3_2var_deg2", "prime" => 3, "prec" => 20,
         "num_vars" => 2, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p3_2var_deg3", "prime" => 3, "prec" => 20,
         "num_vars" => 2, "degree" => 3, "num_samples" => 5),

    # 3 variables
    Dict("name" => "p3_3var_deg1", "prime" => 3, "prec" => 20,
         "num_vars" => 3, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p3_3var_deg2", "prime" => 3, "prec" => 20,
         "num_vars" => 3, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p3_3var_deg3", "prime" => 3, "prec" => 20,
         "num_vars" => 3, "degree" => 3, "num_samples" => 5),

    # ========================================================================
    # GROUP 3: Prime = 5, full degree x variable grid
    # ========================================================================

    # 1 variable
    Dict("name" => "p5_1var_deg1", "prime" => 5, "prec" => 20,
         "num_vars" => 1, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p5_1var_deg2", "prime" => 5, "prec" => 20,
         "num_vars" => 1, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p5_1var_deg3", "prime" => 5, "prec" => 20,
         "num_vars" => 1, "degree" => 3, "num_samples" => 5),

    # 2 variables
    Dict("name" => "p5_2var_deg1", "prime" => 5, "prec" => 20,
         "num_vars" => 2, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p5_2var_deg2", "prime" => 5, "prec" => 20,
         "num_vars" => 2, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p5_2var_deg3", "prime" => 5, "prec" => 20,
         "num_vars" => 2, "degree" => 3, "num_samples" => 5),

    # 3 variables
    Dict("name" => "p5_3var_deg1", "prime" => 5, "prec" => 20,
         "num_vars" => 3, "degree" => 1, "num_samples" => 5),
    Dict("name" => "p5_3var_deg2", "prime" => 5, "prec" => 20,
         "num_vars" => 3, "degree" => 2, "num_samples" => 5),
    Dict("name" => "p5_3var_deg3", "prime" => 5, "prec" => 20,
         "num_vars" => 3, "degree" => 3, "num_samples" => 5),
]

# Use paper experiments by default
experiment_configs = paper_experiments
