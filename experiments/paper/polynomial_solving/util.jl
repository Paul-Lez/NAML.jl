"""
Utility functions for polynomial solving experiments.

Provides:
1. Random polynomial generation with guaranteed roots in the Gauss point (unit ball)
2. Loss function creation for minimizing |f(z)|

Key idea: We construct polynomials that are guaranteed to have a root in Z_p^n
(the unit ball / Gauss point) by building them from factors that vanish at known points.

- 1 variable:  f(x) = (x - r₁)(x - r₂)⋯(x - r_d) with random rᵢ ∈ Z_p
- n variables: f(x₁,…,xₙ) = ∏ᵢ (aᵢ₁x₁ + ⋯ + aᵢₙxₙ - cᵢ)
  where cᵢ = aᵢ₁r₁ + ⋯ + aᵢₙrₙ for a random root r ∈ Z_p^n
"""

using Oscar


"""
    generate_random_padic_integer(K::PadicField, num_terms::Int=8) -> PadicFieldElem

Generate a random p-adic integer (valuation ≥ 0) in K.
"""
function generate_random_padic_integer(K::PadicField, num_terms::Int=8)
    p = Int(Oscar.prime(K))
    prec = Oscar.precision(K)
    return generate_random_padic(p, prec, 0, num_terms)
end


"""
    generate_univariate_polynomial_with_roots(K::PadicField, degree::Int)
    -> (AbsolutePolynomialSum, Vector{PadicFieldElem})

Generate a univariate polynomial of given degree with random roots in Z_p.

Constructs f(x) = (x - r₁)(x - r₂)⋯(x - r_d) where each rᵢ is a random p-adic integer.

# Returns
- `func`: An AbsolutePolynomialSum wrapping the polynomial
- `roots`: The vector of roots [r₁, ..., r_d]
"""
function generate_univariate_polynomial_with_roots(K::PadicField, degree::Int)
    # Generate random roots in Z_p
    roots = [generate_random_padic_integer(K) for _ in 1:degree]

    # Build polynomial ring
    R, (x,) = polynomial_ring(K, ["x"])

    # Construct f(x) = ∏(x - rᵢ)
    poly = R(1)
    for r in roots
        poly *= (x - r)
    end

    func = NAML.AbsolutePolynomialSum([poly])
    return func, roots
end


"""
    generate_multivariate_polynomial_with_root(K::PadicField, num_vars::Int, degree::Int)
    -> (AbsolutePolynomialSum, Vector{PadicFieldElem})

Generate a multivariate polynomial of given degree with a guaranteed root in Z_p^n.

Constructs f(x₁,…,xₙ) = ∏ᵢ₌₁ᵈ lᵢ(x) where each lᵢ is a random linear form
that vanishes at a common random point r ∈ Z_p^n:

    lᵢ(x) = aᵢ₁x₁ + aᵢ₂x₂ + ⋯ + aᵢₙxₙ - (aᵢ₁r₁ + ⋯ + aᵢₙrₙ)

So f(r) = 0 by construction.

# Returns
- `func`: An AbsolutePolynomialSum wrapping the polynomial
- `root`: The vector [r₁, ..., rₙ] ∈ Z_p^n where f vanishes
"""
function generate_multivariate_polynomial_with_root(K::PadicField, num_vars::Int, degree::Int)
    # Generate a random root in Z_p^n
    root = [generate_random_padic_integer(K) for _ in 1:num_vars]

    # Build polynomial ring
    var_names = ["x$i" for i in 1:num_vars]
    R, vars = polynomial_ring(K, var_names)

    # Construct polynomial as product of d random linear forms vanishing at root
    poly = R(1)
    for _ in 1:degree
        # Random coefficients for the linear form
        coeffs = [generate_random_padic_integer(K) for _ in 1:num_vars]

        # Ensure at least one coefficient is a unit (nonzero mod p) to get a nontrivial form
        # Replace first coeff with a unit if all are zero mod p
        p = Int(Oscar.prime(K))
        if all(valuation(c) > 0 for c in coeffs)
            coeffs[1] = K(1 + rand(0:(p-1)))
        end

        # Build linear form: Σ aᵢ xᵢ - Σ aᵢ rᵢ
        linear_form = sum(coeffs[i] * vars[i] for i in 1:num_vars)
        constant = sum(coeffs[i] * root[i] for i in 1:num_vars)
        linear_form -= constant

        poly *= linear_form
    end

    func = NAML.AbsolutePolynomialSum([poly])
    return func, root
end


"""
    generate_polynomial_solving_problem(p::Int, prec::Int, num_vars::Int, degree::Int)
    -> (Loss, Vector{PadicFieldElem})

Generate a polynomial solving problem: minimize |f(z)| where f has a known root.

# Arguments
- `p::Int`: Prime for p-adic field
- `prec::Int`: p-adic precision
- `num_vars::Int`: Number of variables (1, 2, or 3)
- `degree::Int`: Degree of the polynomial (1, 2, or 3)

# Returns
- `loss`: A Loss function computing |f(z)|
- `root`: The known root (for verification)
"""
function generate_polynomial_solving_problem(p::Int, prec::Int, num_vars::Int, degree::Int)
    K = PadicField(p, prec)

    if num_vars == 1
        func, roots = generate_univariate_polynomial_with_roots(K, degree)
        root = roots  # Return all roots
    else
        func, root = generate_multivariate_polynomial_with_root(K, num_vars, degree)
    end

    # Construct polydisc type for typed evaluators
    VP = NAML.ValuationPolydisc{PadicFieldElem, Int, num_vars}

    # Create loss using typed evaluator
    batch_eval = NAML.batch_evaluate_init(func, VP)
    batch_fn = (params) -> map(batch_eval, params)
    loss = NAML.Loss(batch_fn, x -> 0)

    return loss, root
end


"""
    generate_initial_point(num_vars::Int, K::PadicField) -> ValuationPolydisc

Generate the Gauss point (unit ball centered at 0) as starting point.
"""
function generate_initial_point(num_vars::Int, K::PadicField)
    return NAML.ValuationPolydisc(ntuple(i -> K(0), num_vars), ntuple(i -> 0, num_vars))
end
