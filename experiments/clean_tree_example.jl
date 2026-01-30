## Clean Tree Visualization Example
##
## This script creates a single, well-documented tree visualization
## to demonstrate how the loss landscape visualization works.

using Plots
include("../src/NAML.jl")
using .NAML
using Oscar

println("=" ^ 60)
println("LOSS LANDSCAPE TREE VISUALIZATION - CLEAN EXAMPLE")
println("=" ^ 60)

# =============================================================================
# STEP 1: SET UP THE P-ADIC FIELD
# =============================================================================
println("\n--- Step 1: Setting up the 2-adic field ---")

prec = 20  # Precision (number of p-adic digits)
p = 2      # The prime (we work over Q_2, the 2-adic numbers)
K = PadicField(p, prec)

println("Field: Q_$p (the $p-adic numbers)")
println("Precision: $prec digits")

# =============================================================================
# STEP 2: DEFINE THE LEAF POLYDISCS
# =============================================================================
println("\n--- Step 2: Defining leaf polydiscs ---")

# We create 3 polydiscs in 1-dimensional p-adic space.
# Each polydisc is a "ball" in Q_2 defined by:
#   D(center, radius) = {x in Q_2 : v(x - center) >= radius}
# where v is the 2-adic valuation.
#
# IMPORTANT: Higher valuation radius = SMALLER disc
#   - radius=5 means actual size 2^(-5) = 1/32
#   - radius=3 means actual size 2^(-3) = 1/8

# Leaf 1: Centered at 0
d1 = ValuationPolydisc([K(0)], [5])
println("Leaf d1: center = 0, valuation radius = 5 (actual size = 1/32)")

# Leaf 2: Centered at 4 (= 2^2, so v(4) = 2)
d2 = ValuationPolydisc([K(4)], [5])
println("Leaf d2: center = 4, valuation radius = 5 (actual size = 1/32)")

# Leaf 3: Centered at 8 (= 2^3, so v(8) = 3)
d3 = ValuationPolydisc([K(8)], [5])
println("Leaf d3: center = 8, valuation radius = 5 (actual size = 1/32)")

println("\nGeometric interpretation:")
println("  - d1 is a tiny ball around 0")
println("  - d2 is a tiny ball around 4")
println("  - d3 is a tiny ball around 8")
println("  - All three have the same size (radius 5 in valuation)")

# =============================================================================
# STEP 3: BUILD THE CONVEX HULL TREE
# =============================================================================
println("\n--- Step 3: Building the convex hull tree ---")

tree = NAML.convex_hull([d1, d2, d3])

println("The convex hull computes all 'joins' of the input discs.")
println("A join of two discs is the smallest disc containing both.")
println()
println("Tree statistics:")
println("  Total nodes: $(length(tree.nodes))")
println("  Leaf nodes: $(length(tree.leaf_indices)) (the original input discs)")
println("  Internal nodes: $(length(tree.nodes) - length(tree.leaf_indices)) (joins)")

println("\nAll nodes in the tree (sorted by radius):")
sorted_indices = sortperm([sum(tree.nodes[i].radius) for i in 1:length(tree.nodes)])
for i in sorted_indices
    node = tree.nodes[i]
    center_val = Float64(lift(node.center[1]))
    is_leaf = i in tree.leaf_indices ? " <-- LEAF (original input)" : ""
    println("  Node $i: center = $center_val, radius = $(node.radius[1])$is_leaf")
end

# =============================================================================
# STEP 4: UNDERSTAND THE TREE HIERARCHY
# =============================================================================
println("\n--- Step 4: Understanding the tree hierarchy ---")

# Extract the spanning tree structure
positions, spanning_children, spanning_parent, root_idx = NAML.compute_tree_layout(tree)

println("The tree hierarchy is determined by disc containment:")
println("  - A disc D1 is a CHILD of D2 if D1 is contained in D2")
println("  - Smaller discs (higher valuation radius) are deeper in the tree")
println("  - The ROOT is the largest disc (lowest valuation radius)")
println()
println("Root node: $root_idx (radius = $(tree.nodes[root_idx].radius[1]))")
println()
println("Spanning tree structure (parent -> children):")
for (parent, children) in sort(collect(spanning_children))
    if !isempty(children)
        parent_radius = tree.nodes[parent].radius[1]
        children_radii = [tree.nodes[c].radius[1] for c in children]
        println("  Node $parent (radius $parent_radius) -> $children (radii $children_radii)")
    end
end

# =============================================================================
# STEP 5: DEFINE A LOSS FUNCTION
# =============================================================================
println("\n--- Step 5: Defining a loss function ---")

# We define a loss function that depends on the disc's position.
# This creates an interesting landscape where different regions have different loss.
#
# Loss formula: loss = (center - 4)^2 / 10 + |radius - 4| + 1
#   - Minimum loss near center=4 (where d2 is located)
#   - Loss increases as we move away from center=4
#   - Also depends on the disc radius

function loss_function(disc::ValuationPolydisc)
    center_val = Float64(lift(disc.center[1]))
    radius_val = Float64(disc.radius[1])

    # Position term: minimum at center=4
    position_loss = (center_val - 4)^2 / 10

    # Size term: minimum at radius=4
    size_loss = abs(radius_val - 4)

    return position_loss + size_loss + 1.0
end

println("Loss function: L(disc) = (center - 4)^2/10 + |radius - 4| + 1")
println()
println("Loss at each node:")
for i in sorted_indices
    node = tree.nodes[i]
    center_val = Float64(lift(node.center[1]))
    loss = loss_function(node)
    is_leaf = i in tree.leaf_indices ? " (LEAF)" : ""
    println("  Node $i: center=$center_val, radius=$(node.radius[1]), loss=$(round(loss, digits=3))$is_leaf")
end

# =============================================================================
# STEP 6: SAMPLE THE LOSS LANDSCAPE
# =============================================================================
println("\n--- Step 6: Sampling the loss landscape ---")

# Sample loss values along geodesics (paths) between connected nodes
num_samples = 15
landscape = sample_loss_landscape(tree, loss_function, num_samples)

println("Sampled $num_samples points along each edge in the tree.")
println("This captures how the loss varies as we move between discs.")

# =============================================================================
# STEP 7: CREATE THE VISUALIZATION
# =============================================================================
println("\n--- Step 7: Creating the visualization ---")

plt = plot_tree_with_loss(tree, landscape,
                          title="Loss Landscape on 2-adic Tree",
                          colormap=:viridis,
                          line_width=8,
                          node_size=15,
                          figsize=(800, 600))

savefig(plt, "loss_landscape_example.png")
println("Saved: loss_landscape_example.png")

# =============================================================================
# SUMMARY
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("""
The visualization shows:

1. TREE STRUCTURE:
   - Root at TOP (largest disc, smallest valuation radius)
   - Leaves at BOTTOM (smallest discs, largest valuation radius)
   - Each node has exactly one parent (except root)

2. NODE POSITIONS:
   - Y-coordinate: depth in tree (root=high, leaves=low)
   - X-coordinate: spread to avoid overlap

3. EDGE COLORS (viridis colormap):
   - YELLOW/BRIGHT = LOW loss (good)
   - PURPLE/DARK = HIGH loss (bad)

4. INTERPRETATION:
   - The coloring shows where in "parameter space" the loss is low
   - Yellow paths lead to good solutions
   - Purple paths lead to poor solutions
   - Optimization algorithms can use this to navigate the tree

5. NODE MARKERS:
   - WHITE circles = leaves (original input discs)
   - GRAY circles = internal nodes (joins)
""")
