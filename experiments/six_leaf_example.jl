## Loss Landscape with 6 Leaves and Composite Function
##
## Demonstrates a more complex example with:
## - 6 leaf polydiscs
## - Loss function = sum of 3 terms, each composing exp or log with a polynomial

using Plots
include("../src/NAML.jl")
using .NAML
using Oscar

println("=" ^ 60)
println("LOSS LANDSCAPE WITH 6 LEAVES AND COMPOSITE FUNCTION")
println("=" ^ 60)

# Set up the 2-adic field
prec = 20
K = PadicField(2, prec)

println("\n--- Defining 6 leaf polydiscs ---")

# 6 leaves spread across the 2-adic line
d1 = ValuationPolydisc([K(0)], [6])
d2 = ValuationPolydisc([K(4)], [6])
d3 = ValuationPolydisc([K(8)], [6])
d4 = ValuationPolydisc([K(16)], [6])
d5 = ValuationPolydisc([K(32)], [6])
d6 = ValuationPolydisc([K(48)], [6])

leaves = [d1, d2, d3, d4, d5, d6]
for (i, d) in enumerate(leaves)
    c = Float64(lift(d.center[1]))
    println("  d$i: center = $c, radius = $(d.radius[1])")
end

println("\n--- Building convex hull tree ---")
tree = NAML.convex_hull(leaves)
println("Tree has $(length(tree.nodes)) nodes ($(length(tree.leaf_indices)) leaves)")

println("\n--- Defining loss function ---")
println("""
Loss function is a sum of 3 terms:
  term1 = exp(|p₁(x)|/10)     where p₁(x) = (x - 10)²
  term2 = log(1 + |p₂(x)|)    where p₂(x) = (x - 25)²
  term3 = exp(-|p₃(x)|/10)    where p₃(x) = (x - 40)²

Each term is either exp or log composed with an absolute polynomial.
The loss also depends on the disc radius (smaller discs = more precision).
""")

function loss_function(disc::ValuationPolydisc)
    center_val = Float64(lift(disc.center[1]))
    radius_val = Float64(disc.radius[1])

    # Evaluate polynomials at center
    # |p₁(x)| = (x - 10)² normalized
    v1 = (center_val - 10)^2 / 100 + 0.1

    # |p₂(x)| = (x - 25)² normalized
    v2 = (center_val - 25)^2 / 100 + 0.1

    # |p₃(x)| = (x - 40)² normalized
    v3 = (center_val - 40)^2 / 100 + 0.1

    # Compose with exp/log
    term1 = exp(v1 / 5)        # exp composed with |p1|
    term2 = log(1 + v2)        # log composed with |p2|
    term3 = exp(-v3 / 5)       # exp(-x) composed with |p3|

    # Factor in radius (smaller disc = higher precision = slight bonus)
    size_factor = 1.0 + (radius_val - 4) / 20

    return (term1 + term2 + term3) * size_factor
end

println("--- Loss at each node ---")
for i in 1:length(tree.nodes)
    node = tree.nodes[i]
    c = Float64(lift(node.center[1]))
    r = node.radius[1]
    loss = loss_function(node)
    is_leaf = i in tree.leaf_indices ? " (LEAF)" : ""
    println("  Node $i: center=$(round(c, digits=1)), radius=$r, loss=$(round(loss, digits=3))$is_leaf")
end

println("\n--- Sampling loss landscape ---")
landscape = sample_loss_landscape(tree, loss_function, 20)

println("\n--- Creating visualization ---")
plt = plot_tree_with_loss(tree, landscape,
                          title="Loss Landscape (6 leaves, exp/log composite)",
                          colormap=:viridis,
                          line_width=6,
                          node_size=12,
                          figsize=(1000, 700))

savefig(plt, "loss_landscape_6leaves.png")
println("Saved: loss_landscape_6leaves.png")

println("\n--- Tree structure ---")
positions, spanning_children, spanning_parent, root_idx = NAML.compute_tree_layout(tree)
println("Root: Node $root_idx (radius $(tree.nodes[root_idx].radius[1]))")

# Find unique radius levels
radii = sort(unique([sum(tree.nodes[i].radius) for i in 1:length(tree.nodes)]))
println("\nRadius levels (from root to leaves):")
for r in radii
    nodes_at_r = [i for i in 1:length(tree.nodes) if sum(tree.nodes[i].radius) == r]
    leaf_info = [i in tree.leaf_indices ? "$i*" : "$i" for i in nodes_at_r]
    println("  Radius $r: nodes $(join(leaf_info, ", "))  (* = leaf)")
end

println("\nSpanning tree edges:")
for (p, children) in sort(collect(spanning_children))
    if !isempty(children)
        p_r = tree.nodes[p].radius[1]
        c_info = ["$c (r=$(tree.nodes[c].radius[1]))" for c in children]
        println("  Node $p (r=$p_r) -> $(join(c_info, ", "))")
    end
end
