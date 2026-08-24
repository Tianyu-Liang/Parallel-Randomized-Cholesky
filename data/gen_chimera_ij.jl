#=
Generate Laplacians.jl "Chimera" test matrices and write them directly in HYPRE
IJ format (the same format the `ij` driver's -fromfile/-rhsfromfile read), with the
paper-correct RHS  b = M*randn(n) / ||M*randn(n)||  (arXiv:2303.00709, tol target 1e-8).

These are the families where HyPre BoomerAMG-PCG hits "Inf" worst-case (Table 2):
  uni_chimera        : M = lap(a)              (unweighted Laplacian, singular)
  wted_chimera       : M = lap(a)              (weighted Laplacian)
  uni_bndry_chimera  : M returned directly     (unweighted SDDM, PD)
  wted_bndry_chimera : M returned directly     (weighted SDDM, PD)

Usage:
  julia gen_chimera_ij.jl <family> <n> <seed> <out_basepath> [rhs_seed]
    -> writes <out_basepath>.00000 (matrix) and <out_basepath>_rhs.00000 (rhs)

Failures are worst-case over random `seed`, so sweep seed = 1,2,3,... at fixed n.
=#

using Laplacians
using SparseArrays
using LinearAlgebra
using Random
using Printf

function save_ij(base::AbstractString, M::SparseMatrixCSC, b::Vector{Float64})
    n = size(M, 1)
    rows = rowvals(M)
    vals = nonzeros(M)
    open(base * ".00000", "w") do io
        println(io, "0 ", n - 1)          # ilower iupper
        println(io, "0 ", n - 1)          # jlower jupper
        # M is symmetric and stored full in CSC -> emit every stored (i,j,v), 0-indexed
        for j in 1:n
            for k in nzrange(M, j)
                i = rows[k]
                @printf(io, "%d %d %.16e\n", i - 1, j - 1, vals[k])
            end
        end
    end
    open(base * "_rhs.00000", "w") do io
        println(io, "0 ", n - 1)
        for i in 1:n
            @printf(io, "%d %.16e\n", i - 1, b[i])
        end
    end
end

function main()
    length(ARGS) >= 4 || error("usage: gen_chimera_ij.jl <family> <n> <seed> <out_basepath> [rhs_seed]")
    fam   = ARGS[1]
    n     = parse(Int, ARGS[2])
    seed  = parse(Int, ARGS[3])
    base  = ARGS[4]
    rseed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 0

    if fam == "uni_chimera"
        a = uni_chimera(n, seed);        M = lap(a)
    elseif fam == "wted_chimera"
        a = wted_chimera(n, seed);       M = lap(a)
    elseif fam == "uni_bndry_chimera"
        M = uni_bndry_chimera(n, seed)
    elseif fam == "wted_bndry_chimera"
        M = wted_bndry_chimera(n, seed)
    else
        error("unknown family '$fam' (uni_chimera|wted_chimera|uni_bndry_chimera|wted_bndry_chimera)")
    end

    M = SparseMatrixCSC{Float64,Int}(M)
    nrm_n = size(M, 1)
    Random.seed!(rseed)
    b = M * randn(nrm_n)
    nb = norm(b)
    nb == 0 && error("||M g|| == 0; try a different rhs_seed")
    b ./= nb

    save_ij(base, M, b)
    @printf("wrote %s.00000  family=%s n=%d seed=%d nnz=%d\n", base, fam, size(M,1), seed, nnz(M))
end

main()
