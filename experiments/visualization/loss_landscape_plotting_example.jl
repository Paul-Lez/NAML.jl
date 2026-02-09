## Example: Creating Loss Landscape Plots with Plots.jl
##
## This script demonstrates how to use Plots.jl to visualize loss landscapes
## on polydisc trees using the NAML loss landscape visualization tools.
##
## To run this example:
##   1. Install Plots.jl: julia -e 'using Pkg; Pkg.add("Plots")'
##   2. Run: julia experiments/loss_landscape_plotting_example.jl

# Load required packages
using Plots
include("../../src/NAML.jl")
using .NAML
using Oscar

println("=== Loss Landscape Tree Visualization Example ===\n")

# Set up a simple example
prec = 20
K = PadicField(2, prec)
R, (x, a, b) = K["x", "a", "b"]

# Define a polynomial function
g = AbsolutePolynomialSum([(x - a)^2, (x - b)^2])
model = AbstractModel(g, [true, false, false])

# Create polydiscs in parameter space
d1 = ValuationPolydisc([K(0), K(0)], [4, 4])
d2 = ValuationPolydisc([K(0), K(4)], [4, 4])
d3 = ValuationPolydisc([K(4), K(0)], [4, 4])

println("Building convex hull tree...")
tree = NAML.convex_hull([d1, d2, d3])
println("  Tree has $(length(tree.nodes)) nodes")
println("  - $(length(tree.leaf_indices)) leaves")
println("  - $(length(tree.nodes) - length(tree.leaf_indices)) internal nodes\n")

# Define loss function
# Create a simple loss function based on parameter distance from a target
# Target: we want parameters a=1, b=3 (both 2-adic numbers)
target_a = K(1)
target_b = K(3)

function loss_function(param_disc::ValuationPolydisc)
    # Extract parameter values (centers of the polydisc)
    a_center = param_disc.center[1]
    b_center = param_disc.center[2]

    # Compute p-adic distances from targets using NAML.valuation
    # Higher valuation = closer, so we negate for loss
    dist_a = NAML.valuation(a_center - target_a)
    dist_b = NAML.valuation(b_center - target_b)

    # Loss is negative sum of valuations (lower valuation = higher loss)
    # Add radius penalty to make smaller discs have lower loss
    loss = -(Float64(dist_a) + Float64(dist_b)) + sum(collect(param_disc.radius))
    return loss
end

# Sample the landscape
println("Sampling loss landscape...")
landscape = sample_loss_landscape(tree, loss_function, 20)

# ==============================================================================
# Method 1: Tree visualization with loss coloring (NEW - the main feature!)
# ==============================================================================
println("\n=== Method 1: Tree visualization with loss coloring ===")
plt_tree = plot_tree_with_loss(tree, landscape,
                               title="Loss Landscape on Polydisc Tree",
                               colormap=:viridis,
                               line_width=6,
                               node_size=12)
savefig(plt_tree, "tree_loss_landscape.png")
println("Saved: tree_loss_landscape.png")

# ==============================================================================
# Method 2: Plain tree structure (without loss coloring)
# ==============================================================================
println("\n=== Method 2: Plain tree structure ===")
plt_simple = plot_tree_simple(tree,
                              title="Convex Hull Tree Structure",
                              line_width=3,
                              node_size=15)
savefig(plt_simple, "tree_structure.png")
println("Saved: tree_structure.png")

# ==============================================================================
# Method 3: Try different colormaps
# ==============================================================================
println("\n=== Method 3: Different colormaps ===")

colormaps = [:viridis, :plasma, :inferno, :magma]
plots = []

for cmap in colormaps
    p = plot_tree_with_loss(tree, landscape,
                            title=string(cmap),
                            colormap=cmap,
                            line_width=5,
                            node_size=8,
                            show_node_labels=false,
                            figsize=(400, 300))
    push!(plots, p)
end

plt_colormaps = plot(plots..., layout=(2, 2), size=(900, 700),
                     plot_title="Loss Landscape with Different Colormaps")
savefig(plt_colormaps, "tree_colormaps.png")
println("Saved: tree_colormaps.png")

# ==============================================================================
# Method 4: More complex example with more polydiscs
# ==============================================================================
println("\n=== Method 4: More complex tree ===")

# Create more polydiscs for a larger tree
d4 = ValuationPolydisc([K(0), K(8)], [4, 4])
d5 = ValuationPolydisc([K(8), K(0)], [4, 4])
d6 = ValuationPolydisc([K(8), K(8)], [4, 4])

tree_large = NAML.convex_hull([d1, d2, d3, d4, d5, d6])
println("  Larger tree has $(length(tree_large.nodes)) nodes")

landscape_large = sample_loss_landscape(tree_large, loss_function, 15)

plt_large = plot_tree_with_loss(tree_large, landscape_large,
                                title="Larger Loss Landscape Tree",
                                colormap=:plasma,
                                line_width=5,
                                node_size=10,
                                figsize=(1000, 700))
savefig(plt_large, "tree_loss_large.png")
println("Saved: tree_loss_large.png")

# ==============================================================================
# Method 5: Export data and print summary
# ==============================================================================
println("\n=== Method 5: Summary and CSV export ===")
print_landscape_summary(tree, landscape)

export_landscape_csv(tree, landscape, "loss_landscape_data.csv")

# ==============================================================================
# Legacy methods (kept for backwards compatibility)
# ==============================================================================
println("\n=== Legacy: Line plot visualization ===")
plt_legacy = plot_loss_landscape(tree, landscape,
                                 title="Loss vs Geodesic Parameter (Legacy)",
                                 line_width=2.5)
savefig(plt_legacy, "landscape_legacy.png")
println("Saved: landscape_legacy.png")

println("\n=== Visualization Complete ===")
println("\nGenerated files:")
println("  - tree_loss_landscape.png  : Tree with loss-colored edges (MAIN)")
println("  - tree_structure.png       : Plain tree structure")
println("  - tree_colormaps.png       : Different colormap options")
println("  - tree_loss_large.png      : Larger tree example")
println("  - landscape_legacy.png     : Legacy line plot")
println("  - loss_landscape_data.csv  : Raw data export")
