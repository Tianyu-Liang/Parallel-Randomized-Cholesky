#=
Reference comparison: run Laplacians.jl's Approximate Cholesky (approxchol_lap / approxchol_sddm)
on the SAME chimera matrices our pipeline tests, to compare against our AC implementation.

Laplacians.jl applies a GREEDY ordering internally (ApproxCholParams(:deg) = dynamic minimum-degree),
so it isn't sensitive to the input ordering the way our static-ordering driver is. This is the
"source of truth" AC: if it converges where ours fails, the difference is the ordering.

Matrices are regenerated from (family, seed) exactly as gen_chimera_sweep.jl does (deterministic),
so they are the same matrices as the chimera_ij/*.mtx files (UNORDERED — we hand approxchol the raw
adjacency and let it order greedily).

Usage:
  julia compare_laplacians.jl <family> <n> <seed_lo> <seed_hi> [rhs_seed]
    family: uni_chimera | wted_chimera | uni_bndry_chimera | wted_bndry_chimera
  env: TOL (default 1e-8), MAXITS (default 1000), ACORDER (:deg default | :wdeg)
=#
using Laplacians, SparseArrays, LinearAlgebra, Random, Printf

bndry_sddm(L) = (n = size(L, 1); int = setdiff(1:n, 1:ceil(Int, n^(1/3)):n); L[int, int])

# returns (kind, arg, M):  kind=:lap -> arg is adjacency a, M=lap(a);  kind=:sddm -> arg=M=the SDDM matrix
function build(fam, n, s)
    if fam == "uni_chimera"
        a = uni_chimera(n, s);  return (:lap, a, lap(a))
    elseif fam == "wted_chimera"
        a = wted_chimera(n, s); return (:lap, a, lap(a))
    elseif fam == "uni_bndry_chimera"
        M = bndry_sddm(lap(uni_chimera(n, s)));  return (:sddm, M, M)
    elseif fam == "wted_bndry_chimera"
        M = bndry_sddm(lap(wted_chimera(n, s))); return (:sddm, M, M)
    else
        error("unknown family '$fam'")
    end
end

function main()
    length(ARGS) >= 4 || error("usage: compare_laplacians.jl <family> <n> <lo> <hi> [rhs_seed]")
    fam = ARGS[1]; n = parse(Int, ARGS[2]); lo = parse(Int, ARGS[3]); hi = parse(Int, ARGS[4])
    rseed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0
    tol   = parse(Float64, get(ENV, "TOL", "1e-8"))
    maxits = parse(Int, get(ENV, "MAXITS", "1000"))
    acorder = Symbol(get(ENV, "ACORDER", "deg"))
    params = ApproxCholParams(acorder)

    @printf("# Laplacians.jl approxchol (order=:%s, tol=%g, maxits=%d)\n", acorder, tol, maxits)
    @printf("%-34s %10s %8s %14s %9s %9s  %s\n", "matrix", "n", "iters", "relres", "build_s", "solve_s", "verdict")
    println("-"^100)

    for s in lo:hi
        kind, arg, M = build(fam, n, s)
        nn = size(M, 1)
        Random.seed!(rseed + s)
        b = M * randn(nn); b ./= norm(b)

        it = [0]          # length-1 so pcg writes the iteration count into it[1] (Int[] => not recorded)
        local x
        t0 = time()
        if kind == :lap
            f = approxchol_lap(arg; tol=tol, maxits=maxits, params=params, verbose=false)
        else
            f = approxchol_sddm(arg; tol=tol, maxits=maxits, params=params, verbose=false)
        end
        build_s = time() - t0
        t0 = time()
        x = f(b; pcgIts=it, tol=tol, maxits=maxits, verbose=false)
        solve_s = time() - t0

        relres = norm(M * x - b) / norm(b)
        iters = isempty(it) ? -1 : it[1]
        verdict = relres <= 1e3 * tol ? "PASS" : relres <= 1e-2 ? "WEAK" : "FAIL"
        @printf("%-34s %10d %8d %14.3e %9.3f %9.3f  %s\n",
                "$(fam).s$(s)", nn, iters, relres, build_s, solve_s, verdict)
        flush(stdout)
    end
end
main()
