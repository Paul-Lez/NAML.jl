## Demo: Tree Visualization with Varying Loss Values
##
## This script demonstrates the tree visualization with a loss function
## that produces varying values across different polydiscs, showing
## the color mapping in action.

using Plots
include("../src/NAML.jl")
using .NAML
using Oscar

println("=== Tree Visualization Demo with Varying Loss ===\n")

# Set up the p-adic field
prec = 20
K = PadicField(2, prec)

# Create polydiscs at different locations
# Using 1D polydiscs for simplicity
d1 = ValuationPolydisc([K(0)], [5])   # At 0, small disc
d2 = ValuationPolydisc([K(8)], [5])   # At 8, small disc
d3 = ValuationPolydisc([K(16)], [5])  # At 16, small disc
d4 = ValuationPolydisc([K(24)], [5])  # At 24, small disc

println("Building convex hull tree from 4 polydiscs...")
tree = NAML.convex_hull([d1, d2, d3, d4])
println("  Tree has $(length(tree.nodes)) nodes")
println("  - $(length(tree.leaf_indices)) leaves")
println("  - $(length(tree.nodes) - length(tree.leaf_indices)) internal nodes\n")

# Define a loss function that varies based on the disc's center and radius
# This creates an interesting landscape where loss depends on position
function varying_loss(disc::ValuationPolydisc)
    # Get center as a float (lift from p-adic)
    center_val = Float64(lift(disc.center[1]))

    # Get radius (valuation - higher means smaller disc)
    radius_val = Float64(disc.radius[1])

    # Loss function: combination of position and size
    # - Lower loss near center=12 (between 8 and 16)
    # - Higher loss for very small or very large discs
    position_term = (center_val - 12)^2 / 100  # Minimum at center=12
    size_term = abs(radius_val - 3)            # Minimum at radius=3

    return position_term + size_term + 1.0  # Add 1 to keep positive
end

# Sample the landscape
println("Sampling loss landscape...")
landscape = sample_loss_landscape(tree, varying_loss, 15)

# Print summary to see the varying loss values
print_landscape_summary(tree, landscape)

# Create the tree visualization
println("\n=== Creating Tree Visualizations ===\n")

# Main visualization with loss coloring
plt1 = plot_tree_with_loss(tree, landscape,
                           title="Loss Landscape (1D Polydisc Tree)",
                           colormap=:viridis,
                           line_width=8,
                           node_size=15,
                           figsize=(900, 700))
savefig(plt1, "tree_varying_loss_viridis.png")
println("Saved: tree_varying_loss_viridis.png")

# Try plasma colormap (good for showing gradients)
plt2 = plot_tree_with_loss(tree, landscape,
                           title="Loss Landscape (Plasma Colormap)",
                           colormap=:plasma,
                           line_width=8,
                           node_size=15,
                           figsize=(900, 700))
savefig(plt2, "tree_varying_loss_plasma.png")
println("Saved: tree_varying_loss_plasma.png")

# Try thermal colormap (hot=high, cold=low)
plt3 = plot_tree_with_loss(tree, landscape,
                           title="Loss Landscape (Thermal Colormap)",
                           colormap=:thermal,
                           line_width=8,
                           node_size=15,
                           figsize=(900, 700))
savefig(plt3, "tree_varying_loss_thermal.png")
println("Saved: tree_varying_loss_thermal.png")

# Plain structure for comparison
plt4 = plot_tree_simple(tree,
                        title="Tree Structure (No Loss Coloring)",
                        line_width=3,
                        node_size=15,
                        figsize=(900, 700))
savefig(plt4, "tree_structure_1d.png")
println("Saved: tree_structure_1d.png")

println("\n=== Demo Complete ===")
println("\nThe visualizations show:")
println("  - Leaves at the bottom (original input discs)")
println("  - Internal nodes (joins) at higher levels")
println("  - Edge colors indicate loss values")
println("  - Yellow/bright = low loss, Purple/dark = high loss (viridis)")
