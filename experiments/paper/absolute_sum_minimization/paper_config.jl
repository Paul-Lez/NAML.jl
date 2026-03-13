"""
Paper-Ready Experiment Configuration

Comprehensive experiments demonstrating:
1. Effect of problem dimensionality (1D, 2D, 5D)
2. Effect of polynomial degree (linear, quadratic)
3. Effect of optimization degree (1 vs 2)
4. Comparison of optimizers (Random baseline, Greedy, MCTS variants)

Each experiment runs multiple samples to ensure statistical reliability.
"""

# ============================================================================
# PAPER-READY EXPERIMENTS
# ============================================================================

paper_experiments = [
    # ========================================================================
    # GROUP 1: 1D Problems (opt_degree must be 1)
    # ========================================================================
    Dict(
        "name" => "p2_1var_2poly_lin_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p2_1var_2poly_quad_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 2,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p2_1var_3poly_lin_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # ========================================================================
    # GROUP 2: 2D Problems with opt_degree=1
    # ========================================================================
    Dict(
        "name" => "p2_2var_2poly_lin_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p2_2var_3poly_lin_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p2_2var_2poly_quad_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 2,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # ========================================================================
    # GROUP 3: 2D Problems with opt_degree=2 (finer search)
    # ========================================================================
    Dict(
        "name" => "p2_2var_2poly_lin_opt2",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "p2_2var_3poly_lin_opt2",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),

    # ========================================================================
    # GROUP 4: 5D Problems with opt_degree=1 (coarse search)
    # ========================================================================
    Dict(
        "name" => "p2_5var_3poly_lin_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p2_5var_5poly_lin_opt1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # ========================================================================
    # GROUP 5: 5D Problems with opt_degree=2 (fine search) - KEY RESULTS
    # ========================================================================
    Dict(
        "name" => "p2_5var_3poly_lin_opt2",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "p2_5var_5poly_lin_opt2",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),

    # ========================================================================
    # PRIME = 3 EXPERIMENTS (same structure as prime=2)
    # ========================================================================

    # GROUP 1: 1D Problems (opt_degree must be 1)
    Dict(
        "name" => "p3_1var_2poly_lin_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p3_1var_2poly_quad_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 2,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p3_1var_3poly_lin_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # GROUP 2: 2D Problems with opt_degree=1
    Dict(
        "name" => "p3_2var_2poly_lin_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p3_2var_3poly_lin_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p3_2var_2poly_quad_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 2,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # GROUP 3: 2D Problems with opt_degree=2 (finer search)
    Dict(
        "name" => "p3_2var_2poly_lin_opt2",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "p3_2var_3poly_lin_opt2",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),

    # GROUP 4: 5D Problems with opt_degree=1 (coarse search)
    Dict(
        "name" => "p3_5var_3poly_lin_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p3_5var_5poly_lin_opt1",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # GROUP 5: 5D Problems with opt_degree=2 (fine search)
    Dict(
        "name" => "p3_5var_3poly_lin_opt2",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "p3_5var_5poly_lin_opt2",
        "prime" => 3,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),

    # ========================================================================
    # PRIME = 5 EXPERIMENTS (same structure as prime=2)
    # ========================================================================

    # GROUP 1: 1D Problems (opt_degree must be 1)
    Dict(
        "name" => "p5_1var_2poly_lin_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p5_1var_2poly_quad_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 2,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p5_1var_3poly_lin_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # GROUP 2: 2D Problems with opt_degree=1
    Dict(
        "name" => "p5_2var_2poly_lin_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p5_2var_3poly_lin_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p5_2var_2poly_quad_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 2,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # GROUP 3: 2D Problems with opt_degree=2 (finer search)
    Dict(
        "name" => "p5_2var_2poly_lin_opt2",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "p5_2var_3poly_lin_opt2",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),

    # GROUP 4: 5D Problems with opt_degree=1 (coarse search)
    Dict(
        "name" => "p5_5var_3poly_lin_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "p5_5var_5poly_lin_opt1",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),

    # GROUP 5: 5D Problems with opt_degree=2 (fine search)
    Dict(
        "name" => "p5_5var_3poly_lin_opt2",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "p5_5var_5poly_lin_opt2",
        "prime" => 5,
        "prec" => 20,
        "num_polys" => 5,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
]

# Use paper experiments by default
experiment_configs = paper_experiments
