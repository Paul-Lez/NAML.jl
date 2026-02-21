"""
Paper-Ready Experiment Configuration for Polynomial Learning

Comprehensive experiments demonstrating polynomial interpolation across:
1. Multiple polynomial degrees (2, 3, 4, 5)
2. Multiple prime fields (p=2, 3, 5)
3. Comparison of optimizers (Random baseline, Greedy, MCTS variants, DAG-MCTS, DOO)

Each experiment runs 5 samples to ensure statistical reliability.
Total: 12 configurations (4 degrees × 3 primes)
"""

# ============================================================================
# PAPER-READY EXPERIMENTS
# ============================================================================

paper_experiments = [
    # ========================================================================
    # 2-adic experiments
    # ========================================================================
    Dict("name" => "2adic_deg2_3pts", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),
    Dict("name" => "2adic_deg3_4pts", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "2adic_deg4_5pts", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 5, "num_samples" => 5),
    Dict("name" => "2adic_deg5_6pts", "prime" => 2, "prec" => 20,
         "degree" => 5, "n_points" => 6, "num_samples" => 5),

    # ========================================================================
    # 3-adic experiments
    # ========================================================================
    Dict("name" => "3adic_deg2_3pts", "prime" => 3, "prec" => 20,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),
    Dict("name" => "3adic_deg3_4pts", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "3adic_deg4_5pts", "prime" => 3, "prec" => 20,
         "degree" => 4, "n_points" => 5, "num_samples" => 5),
    Dict("name" => "3adic_deg5_6pts", "prime" => 3, "prec" => 20,
         "degree" => 5, "n_points" => 6, "num_samples" => 5),

    # ========================================================================
    # 5-adic experiments
    # ========================================================================
    Dict("name" => "5adic_deg2_3pts", "prime" => 5, "prec" => 20,
         "degree" => 2, "n_points" => 3, "num_samples" => 5),
    Dict("name" => "5adic_deg3_4pts", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 4, "num_samples" => 5),
    Dict("name" => "5adic_deg4_5pts", "prime" => 5, "prec" => 20,
         "degree" => 4, "n_points" => 5, "num_samples" => 5),
    Dict("name" => "5adic_deg5_6pts", "prime" => 5, "prec" => 20,
         "degree" => 5, "n_points" => 6, "num_samples" => 5),
]

# Use paper experiments by default
experiment_configs = paper_experiments
