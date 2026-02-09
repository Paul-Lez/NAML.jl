## Loss Landscape with Cubic Polynomials
##
## Loss = |p₁(x)| + |p₂(x)| + |p₃(x)|
## where each pᵢ is a cubic polynomial (x-a)(x-b)(x-c)
##
## Leaves are polydiscs centered at the roots of these polynomials.

using Plots
include("../src/NAML.jl")
using .NAML
using Oscar

println("=" ^ 60)
println("LOSS LANDSCAPE: SUM OF ABSOLUTE CUBIC POLYNOMIALS")
println("=" ^ 60)

# Set up the 2-adic field
prec = 20
K = PadicField(2, prec)

println("\n--- Defining the cubic polynomials ---")

# Roots for each cubic (9 distinct roots total)
roots_p1 = [K(0), K(2), K(4)]       # p₁(x) = (x-0)(x-2)(x-4)
roots_p2 = [K(8), K(16), K(24)]     # p₂(x) = (x-8)(x-16)(x-24)
roots_p3 = [K(32), K(48), K(64)]    # p₃(x) = (x-32)(x-48)(x-64)

println("p₁(x) = (x - 0)(x - 2)(x - 4)   with roots at 0, 2, 4")
println("p₂(x) = (x - 8)(x - 16)(x - 24) with roots at 8, 16, 24")
println("p₃(x) = (x - 32)(x - 48)(x - 64) with roots at 32, 48, 64")
println("\nLoss function: L(x) = |p₁(x)| + |p₂(x)| + |p₃(x)|")

println("\n--- Creating leaf polydiscs at roots ---")

# Create leaves at all 9 roots with high valuation radius (small discs)
leaf_radius = 8  # High valuation = small disc
all_roots = vcat(roots_p1, roots_p2, roots_p3)
leaves = [ValuationPolydisc([r], [leaf_radius]) for r in all_roots]

for (i, leaf) in enumerate(leaves)
    c = Float64(lift(leaf.center[1]))
    println("  Leaf $i: center = $c, radius = $(leaf.radius[1])")
end

println("\n--- Building convex hull tree ---")
tree = NAML.convex_hull(leaves)
println("Tree has $(length(tree.nodes)) nodes ($(length(tree.leaf_indices)) leaves)")

println("\n--- Defining loss function ---")

# Loss function: sum of absolute values of the three cubics
function loss_function(disc::ValuationPolydisc)
    x = Float64(lift(disc.center[1]))

    # p₁(x) = (x-0)(x-2)(x-4) = x(x-2)(x-4)
    p1 = abs(x * (x - 2) * (x - 4))

    # p₂(x) = (x-8)(x-16)(x-24)
    p2 = abs((x - 8) * (x - 16) * (x - 24))

    # p₃(x) = (x-32)(x-48)(x-64)
    p3 = abs((x - 32) * (x - 48) * (x - 64))

    # Normalize to reasonable scale (divide by a constant)
    loss = (p1 + p2 + p3) / 1000

    # Add small term based on radius to break ties
    loss += disc.radius[1] / 100

    return loss
end

println("\n--- Loss at each leaf (root) ---")
println("At a root of pᵢ, that term is 0, but other terms contribute.\n")

for i in tree.leaf_indices
    node = tree.nodes[i]
    c = Float64(lift(node.center[1]))
    loss = loss_function(node)

    # Identify which polynomial has this as a root
    which_poly = if c in [0, 2, 4]
        "root of p₁"
    elseif c in [8, 16, 24]
        "root of p₂"
    else
        "root of p₃"
    end

    println("  Leaf $i (center=$c): loss = $(round(loss, digits=4)) — $which_poly")
end

println("\n--- Sampling loss landscape ---")
landscape = sample_loss_landscape(tree, loss_function, 20)

println("\n--- Creating visualization ---")
plt = plot_tree_with_loss(tree, landscape,
                          title="Loss = |p₁(x)| + |p₂(x)| + |p₃(x)| (cubics)",
                          colormap=:viridis,
                          line_width=5,
                          node_size=10,
                          figsize=(1200, 800))

savefig(plt, "loss_landscape_cubics.png")
println("Saved: loss_landscape_cubics.png")

println("\n--- Tree structure by radius level ---")
radii = sort(unique([tree.nodes[i].radius[1] for i in 1:length(tree.nodes)]))
println("Radius levels (root at top, leaves at bottom):")
for r in radii
    nodes_at_r = [i for i in 1:length(tree.nodes) if tree.nodes[i].radius[1] == r]
    centers = [round(Float64(lift(tree.nodes[i].center[1])), digits=0) for i in nodes_at_r]
    is_leaf = r == leaf_radius ? " (LEAVES)" : ""
    println("  Radius $r: $(length(nodes_at_r)) nodes at centers $centers$is_leaf")
end

println("\n" * "=" ^ 60)
println("INTERPRETATION")
println("=" ^ 60)
println("""
The loss is minimized at the roots of the polynomials:
- Near roots of p₁ (0, 2, 4): p₁ term → 0, loss from p₂ and p₃
- Near roots of p₂ (8, 16, 24): p₂ term → 0, loss from p₁ and p₃
- Near roots of p₃ (32, 48, 64): p₃ term → 0, loss from p₁ and p₂

The "best" roots (lowest total loss) depend on the balance of
contributions from the other two polynomials.

Yellow paths indicate low-loss regions (near polynomial roots).
Purple paths indicate high-loss regions (far from all roots).
""")
