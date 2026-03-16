"""
Compare gradient descent vs greedy descent on |f(x)| minimization over Q_p.

For each random univariate polynomial f with known roots in Z_p, both optimizers
try to minimize |f(x)| from the same starting point (the Gauss point B(0, p^0) = Z_p).

A "divergence" is flagged whenever the two optimizers reach qualitatively different
outcomes (one finds the root, the other doesn't) or their final losses differ by
more than DIV_RATIO.

Usage:
    julia --project=. experiments/gradient_vs_greedy.jl
"""

include("../src/NAML.jl")
using .NAML
using Oscar
using Printf
using Random
using LinearAlgebra

Random.seed!(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

"""Generate a random p-adic integer in K with up to `num_terms` base-p digits."""
function rand_padic(K::PadicField, num_terms::Int = 8)
    p   = Int(Oscar.prime(K))
    acc = K(0)
    for i in 0:(num_terms - 1)
        d = rand(0:(p - 1))
        d == 0 && continue
        acc += K(d) * K(p)^i
    end
    return acc
end

"""Create loss + grad closures for minimizing |f(x)|."""
function make_loss(f::AbsolutePolynomialSum, ::Type{VP}) where VP
    batch_eval = NAML.batch_evaluate_init(f, VP)
    Loss(
        params   -> map(batch_eval, params),
        tangents -> [directional_derivative(batch_eval, t) for t in tangents]
    )
end

"""
Run optimizer for n_steps, returning:
  - final (last valid) loss
  - best (lowest) loss seen
  - param at best loss
  - loss trajectory (length n_steps+1, index 1 = initial; padded with last value if converged early)
"""
function run_optim!(optim, n_steps::Int)
    traj       = fill(0.0, n_steps + 1)
    traj[1]    = eval_loss(optim)
    best_loss  = traj[1]
    best_param = optim.param
    last_i     = 0
    for i in 1:n_steps
        optim.converged && break
        step!(optim)
        l          = eval_loss(optim)
        traj[i+1]  = l
        last_i     = i
        if l < best_loss
            best_loss  = l
            best_param = optim.param
        end
    end
    # pad any unwritten tail with the last known value
    last_val = last_i == 0 ? traj[1] : traj[last_i + 1]
    for i in (last_i + 2):(n_steps + 1)
        traj[i] = last_val
    end
    return last_val, best_loss, best_param, traj
end

# ── Configuration ─────────────────────────────────────────────────────────────

const PRIMES    = [2, 3, 5]
const DEGREES   = [1, 2, 3]
const N_TRIALS  = 30          # trials per (prime, degree) combination
const N_STEPS   = 50          # optimization steps per trial
const PREC      = 20          # p-adic precision
const EPS       = 1e-10       # "found root" threshold
const DIV_RATIO = 100.0       # flag if best losses differ by more than this factor

# ── Main loop ─────────────────────────────────────────────────────────────────

println("Gradient descent vs greedy descent: |f(x)| minimization over Q_p")
println("="^70)
@printf("  Trials per (p, degree): %d   Steps: %d   Precision: %d\n",
        N_TRIALS, N_STEPS, PREC)
println("="^70, "\n")

struct DivRecord
    p::Int; degree::Int; trial::Int
    roots::Vector
    poly::Any
    greedy_final::Float64;   greedy_best::Float64;   greedy_param::Any
    gradient_final::Float64; gradient_best::Float64; gradient_param::Any
    greedy_traj::Vector{Float64}
    gradient_traj::Vector{Float64}
end

records = DivRecord[]

for p in PRIMES, degree in DEGREES
    K  = PadicField(p, PREC)
    VP = ValuationPolydisc{PadicFieldElem, Int, 1}

    n_div = 0

    for trial in 1:N_TRIALS
        # ── Build a random polynomial with known roots in Z_p ──────────────
        R, (x,) = polynomial_ring(K, ["x"])
        roots   = [rand_padic(K) for _ in 1:degree]
        poly    = prod(x - r for r in roots)
        f       = AbsolutePolynomialSum([poly])
        loss    = make_loss(f, VP)

        # Gauss point B(0, p^0) = Z_p: contains all roots by construction
        init = VP((K(0),), (0,))

        greedy_optim   = greedy_descent_init(  init, loss, 1, (false, 1))
        gradient_optim = gradient_descent_init(init, loss, 1, (false, 1))

        gf, gb, gp, gt   = run_optim!(greedy_optim,   N_STEPS)
        ddf, ddb, ddp, ddt = run_optim!(gradient_optim, N_STEPS)

        # ── Divergence check ───────────────────────────────────────────────
        greedy_found   = gb  < EPS
        gradient_found = ddb < EPS

        qualitative_diff = greedy_found != gradient_found
        ratio_diff = max(gb, ddb) / max(min(gb, ddb), 1e-300) > DIV_RATIO

        if qualitative_diff || ratio_diff
            n_div += 1
            push!(records, DivRecord(p, degree, trial, roots, poly,
                                     gf, gb, gp, ddf, ddb, ddp, gt, ddt))
        end
    end

    @printf("  p=%d  degree=%d  divergences: %2d / %d\n",
            p, degree, n_div, N_TRIALS)
end

# ── Summary ───────────────────────────────────────────────────────────────────

println()
println("="^70)
@printf("Total divergences: %d / %d\n",
        length(records), length(PRIMES) * length(DEGREES) * N_TRIALS)
println("="^70)

if isempty(records)
    println("\nNo divergences found — both optimizers always agree.")
else
    println("\nDIVERGENCE DETAILS\n")
    for r in records
        println("-"^70)
        @printf("p=%d  degree=%d  trial=%d\n", r.p, r.degree, r.trial)
        println("  poly:  ", r.poly)
        println("  roots: ", join(string.(r.roots), ",  "))
        @printf("  greedy   best=%.3e  final=%.3e  radius=%d\n",
                r.greedy_best, r.greedy_final, NAML.radius(r.greedy_param)[1])
        @printf("  gradient best=%.3e  final=%.3e  radius=%d\n",
                r.gradient_best, r.gradient_final, NAML.radius(r.gradient_param)[1])
        # Show first 10 steps of each trajectory
        n_show = min(10, length(r.greedy_traj))
        print("  greedy   traj: ")
        println(join([@sprintf("%.2e", v) for v in r.greedy_traj[1:n_show]], "  "))
        print("  gradient traj: ")
        println(join([@sprintf("%.2e", v) for v in r.gradient_traj[1:n_show]], "  "))
    end
    println("-"^70)
end

println("\nDone!")
