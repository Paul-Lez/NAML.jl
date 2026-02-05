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
        "name" => "1D_2poly_linear_deg1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "1D_2poly_quadratic_deg1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 1,
        "degree" => 2,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "1D_3poly_linear_deg1",
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
        "name" => "2D_2poly_linear_deg1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "2D_3poly_linear_deg1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "2D_2poly_quadratic_deg1",
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
        "name" => "2D_2poly_linear_deg2",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 2,
        "num_vars" => 2,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "2D_3poly_linear_deg2",
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
        "name" => "5D_3poly_linear_deg1",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 1
    ),
    Dict(
        "name" => "5D_5poly_linear_deg1",
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
        "name" => "5D_3poly_linear_deg2",
        "prime" => 2,
        "prec" => 20,
        "num_polys" => 3,
        "num_vars" => 5,
        "degree" => 1,
        "num_samples" => 5,
        "opt_degree" => 2
    ),
    Dict(
        "name" => "5D_5poly_linear_deg2",
        "prime" => 2,
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
