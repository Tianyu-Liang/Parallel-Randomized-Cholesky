#=
Prepare the Flow-IPM matrices for the AC-vs-HyPre head-to-head.

The ipmMat/*.mm files are RAW, UNORDERED, singular graph Laplacians (verified: row sums ~0 to machine
precision -- the IPM `eps` only shrinks edge weights, it does NOT make them SDDM). So AC factorizes
them in plain GRAPH mode (is_graph=1), like the `*_chimera` family -- no boundary/augmentation.

This script reads each .mm, applies an AMD fill-reducing ordering (AC is ordering-sensitive; this
matches the chimera_amd pipeline), generates the paper RHS b = M*g/||M*g||, and writes the standard
3-file layout into <outdir>:
    <base>.mtx                 -> symmetric lower-tri MatrixMarket, for the AC driver (graph mode)
    <base>.00000 + _rhs.00000  -> HyPre IJ (full matrix + rhs)
so that `run_ac_vs_hypre.sh <outdir>` compares AC vs HyPre on them (HyPre 1-thread/1000-iter, AC AMD).

Because the names start with `uni_chimera` (no "bndry"), run_ac_vs_hypre.sh auto-selects GRAPH mode.

Usage:
  julia gen_ipm_amd.jl <outdir> <mm-file> [<mm-file> ...]        (env RSEED, default 0)
=#
using MatrixMarket, SparseArrays, LinearAlgebra, Random, Printf, AMD

function save_ij(base, M::SparseMatrixCSC, b::Vector{Float64})
    n = size(M, 1); rows = rowvals(M); vals = nonzeros(M)
    open(base * ".00000", "w") do io
        println(io, "0 ", n - 1); println(io, "0 ", n - 1)
        for j in 1:n, k in nzrange(M, j)
            @printf(io, "%d %d %.16e\n", rows[k] - 1, j - 1, vals[k])
        end
    end
    open(base * "_rhs.00000", "w") do io
        println(io, "0 ", n - 1)
        for i in 1:n; @printf(io, "%d %.16e\n", i - 1, b[i]); end
    end
end

function save_mtx(base, M::SparseMatrixCSC)
    n = size(M, 1); rows = rowvals(M); vals = nonzeros(M)
    nnz_lo = 0
    for j in 1:n, k in nzrange(M, j); rows[k] >= j && (nnz_lo += 1); end
    open(base * ".mtx", "w") do io
        println(io, "%%MatrixMarket matrix coordinate real symmetric")
        @printf(io, "%d %d %d\n", n, n, nnz_lo)
        for j in 1:n, k in nzrange(M, j)
            rows[k] >= j && @printf(io, "%d %d %.16e\n", rows[k], j, vals[k])
        end
    end
end

function main()
    length(ARGS) >= 2 || error("usage: gen_ipm_amd.jl <outdir> <mm-file> [<mm-file> ...]")
    outdir = ARGS[1]; mmfiles = ARGS[2:end]
    rseed = parse(Int, get(ENV, "RSEED", "0"))
    mkpath(outdir)
    for (idx, mm) in enumerate(mmfiles)
        isfile(mm) || (println("missing: $mm"); continue)
        M = SparseMatrixCSC{Float64,Int}(MatrixMarket.mmread(mm))
        name = replace(basename(mm), ".mm" => "")
        p = amd(M); M = M[p, p]                          # AMD fill-reducing order
        Random.seed!(rseed + idx)                        # deterministic per-file b
        b = M * randn(size(M, 1)); nb = norm(b)
        nb == 0 && (println("$name: ||Mg||=0, skip"); continue)
        b ./= nb
        base = joinpath(outdir, name)
        save_ij(base, M, b); save_mtx(base, M)
        @printf("%-44s n=%d nnz=%d -> AMD .mtx + HyPre IJ\n", name, size(M, 1), nnz(M)); flush(stdout)
    end
end
main()
