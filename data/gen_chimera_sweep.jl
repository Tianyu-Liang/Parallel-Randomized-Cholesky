#=
Batch-generate a seed sweep of one Chimera family into HYPRE IJ files, in a single
Julia process (amortizes the ~20s compile). RHS = M*randn / ||.|| (paper recipe).

Usage:
  julia gen_chimera_sweep.jl <family> <n> <seed_lo> <seed_hi> <outdir> [rhs_seed]
    family: uni_chimera | wted_chimera | uni_bndry_chimera | wted_bndry_chimera
  -> per seed: <base>.00000 + <base>_rhs.00000  (HYPRE ij)
              <base>.mtx                        (MatrixMarket, for the AC driver ./driver)

=========================== IMPORTANT: SDDM vs Laplacian =============================
The two output formats are NOT always the same matrix:

  * HYPRE (.00000) always gets the RAW matrix M -- a Laplacian for the *_chimera
    families, an SDDM (boundary-removed, nonsingular) for the *_bndry_chimera families.
    HYPRE/AMG treat any SPD matrix directly, so the raw matrix is what they need.

  * The AC driver (.mtx) is different for the SDDM families. Our AC factorizes a
    *Laplacian*; an SDDM matrix M (nonzero row sums / "excess" diagonal) is NOT a
    Laplacian, and feeding it to the AC driver in GRAPH mode silently produces a
    DEGRADED preconditioner (it mishandles the excess diagonal). The driver also does
    NOT auto-augment -- PHYSICS mode (is_graph=0) only TRIMS an already-appended ground
    row/col. So for the SDDM families we write the AUGMENTED Laplacian to .mtx: M plus a
    single ground node (index n+1, holding the per-row excess) so that row sums are 0.
    Then the AC driver MUST be run in PHYSICS mode (is_graph=0), which eliminates and
    then trims the ground node, leaving a factor G with G^T G ~= M. See README.

  Net: *_bndry_chimera .mtx files are size n+1 (augmented) and require `./driver ... 0`;
       *_chimera        .mtx files are size n   (raw Laplacian)  and require `./driver ... 1`.
       The _rhs.00000 always has n entries (the size of the SOLVED system, i.e. M).
=====================================================================================
=#
using Laplacians, SparseArrays, LinearAlgebra, Random, Printf, AMD

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

# symmetric half-stored MatrixMarket (lower triangle incl. diagonal) — the format the
# AC driver's fast_matrix_market reader consumes (same as the ipmMat *.mm files).
function save_mtx(base, M::SparseMatrixCSC)
    n = size(M, 1); rows = rowvals(M); vals = nonzeros(M)
    nnz_lo = 0
    for j in 1:n, k in nzrange(M, j); rows[k] >= j && (nnz_lo += 1); end
    open(base * ".mtx", "w") do io
        println(io, "%%MatrixMarket matrix coordinate real symmetric")
        @printf(io, "%d %d %d\n", n, n, nnz_lo)
        for j in 1:n, k in nzrange(M, j)
            rows[k] >= j && @printf(io, "%d %d %.16e\n", rows[k], j, vals[k])  # 1-indexed
        end
    end
end

# SDDM (nonsingular) "boundary" variant: drop ~n^(1/3) boundary nodes from the Laplacian, which
# makes the remaining rows strictly diagonally dominant -> SPD/nonsingular. Replicates Laplacians.jl
# uni/wted_bndry_chimera, but with the ceil(Int,...) fix the installed 1.4.1 lacks (it uses
# ceil(n^(1/3)) -> Float index -> "invalid index: 2.0" on Julia 1.12; fixed only on upstream master).
# Returns (M, excess): M = L[int,int] (the SDDM), excess[i] = total weight from interior node i to
# the REMOVED nodes (= row sum of M, >= 0, EXACT -- summed from same-sign entries, no cancellation).
function bndry_sddm_excess(L)
    n = size(L, 1)
    removed = collect(1:ceil(Int, n^(1/3)):n)
    int = setdiff(1:n, removed)
    M = L[int, int]
    excess = -vec(sum(L[int, removed], dims = 2))   # L's interior->removed entries are <0 -> excess >= 0
    return M, excess
end

# Augment a raw SDDM M (size k) into a true Laplacian L (size k+1) by appending ONE ground node
# (index k+1) that holds the excess: L = [ M  -e ; -e'  sum(e) ], so every row sums to 0. The ground
# node is LAST so PHYSICS mode (which trims the last row/col) recovers G^T G ~= M. Only interior nodes
# with real excess (adjacent to a removed boundary node) connect to ground -> ground degree stays small.
function augment_laplacian(M::SparseMatrixCSC, excess::Vector{Float64})
    k = size(M, 1); rows = rowvals(M); vals = nonzeros(M)
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:k, t in nzrange(M, j)
        push!(I, rows[t]); push!(J, j); push!(V, vals[t])
    end
    g = k + 1; ssum = 0.0
    for i in 1:k
        if excess[i] != 0
            push!(I, i); push!(J, g); push!(V, -excess[i])
            push!(I, g); push!(J, i); push!(V, -excess[i])
            ssum += excess[i]
        end
    end
    push!(I, g); push!(J, g); push!(V, ssum)
    return sparse(I, J, V, g, g)
end

base_lap(fam, n, s) =
    fam == "uni_chimera"        ? lap(uni_chimera(n, s))  :
    fam == "wted_chimera"       ? lap(wted_chimera(n, s)) :
    fam == "uni_bndry_chimera"  ? lap(uni_chimera(n, s))  :
    fam == "wted_bndry_chimera" ? lap(wted_chimera(n, s)) :
    error("unknown family '$fam'")

# nnz-sort ordering (same recipe as cpu_implementation/write_graph.jl `graph_share`): random
# tie-break shuffle, then sort columns by ascending nnz (~minimum-degree). Returns the COMPOSITE
# permutation q so that M[q,q] is the ordered matrix (and excess[q] keeps the excess vector aligned).
# AC's incomplete-Cholesky quality is ordering-sensitive; this is the ordering the AC pipeline uses.
function nnz_sort_perm(M)
    n = size(M, 1)
    p0 = randperm(n); M0 = M[p0, p0]                      # random tie-break shuffle
    nnz_col = [M0.colptr[i + 1] - M0.colptr[i] for i in 1:n]
    p = sortperm(nnz_col)
    return p0[p]                                          # M[q,q] == nnz_sort(M),  q = p0[p]
end

function main()
    length(ARGS) >= 5 || error("usage: gen_chimera_sweep.jl <family> <n> <lo> <hi> <outdir> [rhs_seed]  (env CHIMERA_ORDER=none|nnz|amd)")
    fam = ARGS[1]; n = parse(Int, ARGS[2]); lo = parse(Int, ARGS[3]); hi = parse(Int, ARGS[4])
    outdir = ARGS[5]; rseed = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0
    order = get(ENV, "CHIMERA_ORDER", "none")             # CHIMERA_ORDER=nnz|amd -> reorder the matrix
    is_sddm = occursin("bndry", fam)                      # *_bndry_chimera are SDDM -> augment for AC
    mkpath(outdir)
    for s in lo:hi
        if is_sddm
            M, excess = bndry_sddm_excess(base_lap(fam, n, s))
            M = SparseMatrixCSC{Float64,Int}(M)
        else
            M = SparseMatrixCSC{Float64,Int}(base_lap(fam, n, s)); excess = Float64[]
        end
        Random.seed!(rseed + s)
        # Ordering is applied to the SOLVED matrix M (and excess kept in step). For SDDM the ground
        # node is appended AFTER ordering so it always stays last (required by physics-mode trim).
        if order == "nnz"
            q = nnz_sort_perm(M); M = M[q, q]; is_sddm && (excess = excess[q])
        elseif order == "amd"
            q = amd(M); M = M[q, q]; is_sddm && (excess = excess[q])
        end
        b = M * randn(size(M, 1)); nb = norm(b)
        nb == 0 && (println("seed $s: ||Mg||=0, skip"); continue)
        b ./= nb
        base = joinpath(outdir, "$(fam).n$(n).s$(s)")
        save_ij(base, M, b)                               # HYPRE: raw matrix (size k)
        if is_sddm
            L = augment_laplacian(M, excess)              # AC: augmented Laplacian (size k+1, ground last)
            save_mtx(base, L)
            gdeg = L.colptr[size(L,1)+1] - L.colptr[size(L,1)]   # ground-node column nnz (sanity)
            @printf("seed %d: SDDM n=%d -> AC .mtx augmented to %d (ground_deg=%d, PHYSICS mode is_graph=0); HYPRE .00000 raw n=%d; order=%s\n",
                    s, size(M,1), size(L,1), gdeg - 1, size(M,1), order)
        else
            save_mtx(base, M)                             # AC: raw Laplacian (size n, GRAPH mode is_graph=1)
            @printf("seed %d: LAP n=%d nnz=%d (AC GRAPH mode is_graph=1); order=%s\n", s, size(M,1), nnz(M), order)
        end
        flush(stdout)
    end
end
main()
