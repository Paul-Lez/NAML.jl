"""
Paper-Ready Experiment Configuration for Function Learning

Comprehensive experiments demonstrating binary classification learning across:
1. Three target functions (zero, one, random)
2. Multiple polynomial degrees (2, 3, 4)
3. Multiple prime fields (p=2, 3, 5)
4. Comparison of optimizers (Random baseline, Greedy variants, MCTS variants, DAG-MCTS, DOO)

Each experiment runs 5 samples to ensure statistical reliability.
Total: 27 configurations (3 functions × 3 degrees × 3 primes)
"""

# ============================================================================
# PAPER-READY EXPERIMENTS
# ============================================================================

paper_experiments = [
    # ========================================================================
    # 2-adic experiments
    # ========================================================================

    # Zero function - 2-adic
    Dict("name" => "zero_fn_deg2_2adic", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg4_2adic", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 2-adic
    Dict("name" => "one_fn_deg2_2adic", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg4_2adic", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Random function - 2-adic
    Dict("name" => "random_fn_deg2_2adic", "prime" => 2, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "random_fn_deg3_2adic", "prime" => 2, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "random_fn_deg4_2adic", "prime" => 2, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # ========================================================================
    # 3-adic experiments
    # ========================================================================

    # Zero function - 3-adic
    Dict("name" => "zero_fn_deg2_3adic", "prime" => 3, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_3adic", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg4_3adic", "prime" => 3, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 3-adic
    Dict("name" => "one_fn_deg2_3adic", "prime" => 3, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg3_3adic", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg4_3adic", "prime" => 3, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Random function - 3-adic
    Dict("name" => "random_fn_deg2_3adic", "prime" => 3, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "random_fn_deg3_3adic", "prime" => 3, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "random_fn_deg4_3adic", "prime" => 3, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # ========================================================================
    # 5-adic experiments
    # ========================================================================

    # Zero function - 5-adic
    Dict("name" => "zero_fn_deg2_5adic", "prime" => 5, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg3_5adic", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "zero_fn_deg4_5adic", "prime" => 5, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "zero",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # One function - 5-adic
    Dict("name" => "one_fn_deg2_5adic", "prime" => 5, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg3_5adic", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "one_fn_deg4_5adic", "prime" => 5, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "one",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),

    # Random function - 5-adic
    Dict("name" => "random_fn_deg2_5adic", "prime" => 5, "prec" => 20,
         "degree" => 2, "n_points" => 4, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "random_fn_deg3_5adic", "prime" => 5, "prec" => 20,
         "degree" => 3, "n_points" => 5, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
    Dict("name" => "random_fn_deg4_5adic", "prime" => 5, "prec" => 20,
         "degree" => 4, "n_points" => 6, "target_fn" => "random",
         "num_samples" => 5, "threshold" => 0.5, "scale" => 1.0),
]

# Use paper experiments by default
experiment_configs = paper_experiments
