"""
Sanity Check Configuration: MCTS vs DAG-MCTS

Tests MCTS and DAG-MCTS performance across:
- Different simulation counts: 50, 100, 200
- Different opt_degree parameters: 1, 2, dimension-matched
- Representative problem instances

For absolute sum minimization, dimension = num_vars
"""

sanity_experiments = [
    # 1D problem (dimension = 1)
    Dict("name" => "sanity_1D_2poly", "prime" => 2, "prec" => 20,
         "num_polys" => 2, "num_vars" => 1, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),

    # 2D problem (dimension = 2)
    Dict("name" => "sanity_2D_3poly", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 2, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),

    # 5D problem (dimension = 5)
    Dict("name" => "sanity_5D_3poly", "prime" => 2, "prec" => 20,
         "num_polys" => 3, "num_vars" => 5, "degree" => 1, "num_samples" => 5, "opt_degree" => 1),
]

# Use sanity check experiments
experiment_configs = sanity_experiments
