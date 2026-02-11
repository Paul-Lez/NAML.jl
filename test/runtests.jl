# Main test runner for NAML package
#
# This file is the entry point for the test suite. It can be run with:
#   julia --project test/runtests.jl
# or via Pkg.test():
#   using Pkg; Pkg.test("NAML")

using Test
using NAML

@testset "NAML.jl" begin
    @testset "Basic Structures" begin
        include("valued_point.jl")
        include("polydisc.jl")
        include("tangent_vector.jl")
        include("functions.jl")
        include("test_typed_evaluators.jl")
    end

    @testset "Statistics" begin
        include("frechet.jl")
        include("least_squares.jl")
    end

    @testset "Optimization" begin
        include("gradient_descent.jl")
        include("polynomial_learning.jl")
        include("bivariate_optimization.jl")
        include("linear_optimization.jl")
    end

    @testset "Tree Search Algorithms" begin
        include("dag_mcts.jl")
        include("test_doo.jl")
        include("test_doo_3d.jl")
        include("test_all_optimizers.jl")
    end

    @testset "Visualization" begin
        include("convex_hull.jl")
        include("geodesic.jl")
        include("loss_landscape.jl")
        include("loss_landscape_visualization.jl")
    end
end

println("\n✓ All tests completed successfully!")
