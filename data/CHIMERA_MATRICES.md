# Chimera Test Matrices

This document describes the Chimera benchmark matrices used in the head-to-head
experiments (`run_ac_vs_hypre.sh`), how they are generated, and what the four
families mean.

## 1. What these matrices are

Each matrix is a sparse symmetric linear system `M x = b` where `M` is either a
**graph Laplacian** or an **SDDM matrix** (Symmetric Diagonally Dominant with
non-positive off-diagonals — i.e. a graph Laplacian with some rows made strictly
dominant). They come from the **"chimera" generator** in
[`Laplacians.jl`](https://github.com/danspielman/Laplacians.jl) (Spielman & Kyng),
which is the **SDDM2023** benchmark introduced in:

> Yuan Gao, Rasmus Kyng, Daniel A. Spielman,
> *"Robust and Practical Solution of Laplacian Equations by Approximate Elimination,"*
> arXiv:2303.00709.

The chimera generator is **purpose-built to stress-test solver robustness**: rather
than one matrix type, it is a randomized generator that assembles structurally
diverse graphs (grids, paths, trees, expanders, random graphs, glued by products and
joins). Each seed draws a different structure spanning the spectrum from trivially
easy to numerically pathological. This is why per-matrix iteration counts and solver
failures vary so much across seeds.

## 2. The four families at a glance

All families are generated at **`n = 100000`** vertices and **seeds `s = 1, …, 50`**.

| Family                 | Edge weights | Boundary | Resulting matrix `M`              | Definiteness          | Solved size |
|------------------------|--------------|----------|-----------------------------------|-----------------------|-------------|
| `uni_chimera`          | unweighted   | no       | graph Laplacian `L = D − A`       | singular (PSD)        | 100000      |
| `wted_chimera`         | weighted     | no       | weighted graph Laplacian          | singular (PSD)        | 100000      |
| `uni_bndry_chimera`    | unweighted   | **yes**  | SDDM (`L` with boundary removed)  | **nonsingular (PD)**  | 97872       |
| `wted_bndry_chimera`   | weighted     | **yes**  | SDDM (weighted, boundary removed) | **nonsingular (PD)**  | 97872       |

There are two independent binary axes: **weighted vs. unweighted** and
**boundary vs. no-boundary**.

## 3. Boundary vs. no-boundary (the key label)

This is the most important distinction.

### No-boundary (`*_chimera`) — singular graph Laplacian
The matrix is the full **graph Laplacian** `L = D − A` of the chimera graph
(`D` = degree diagonal, `A` = adjacency). A graph Laplacian:
- has **row sums exactly 0** (each diagonal equals the sum of off-diagonals),
- is **symmetric positive semi-definite but singular** — the all-ones vector is in
  its null space (constant potentials), so `L` is rank `n−1` for a connected graph.

This is the "pure graph problem." The system `L x = b` is solvable only for `b`
orthogonal to the null space (which `b = L·g` guarantees).

### Boundary (`*_bndry_chimera`) — nonsingular SDDM
Starting from the same Laplacian `L`, a fixed set of **"boundary" vertices is
removed** — their rows and columns are deleted — and we keep the principal submatrix
`M = L[interior, interior]`. Removing those nodes is equivalent to imposing a
**homogeneous Dirichlet (grounding) boundary condition** on them: every interior
vertex that had an edge to a removed vertex now has a row whose diagonal **strictly
exceeds** the sum of its remaining off-diagonals. The result is an **SDDM matrix**:
symmetric, **positive definite, nonsingular**, and uniquely solvable. This is the
"boundary-value / physics" version of the same graph.

The removed set is the **deterministic, stride-`⌈n^{1/3}⌉` index set**
`1 : ⌈n^{1/3}⌉ : n` (the chimera graph has no geometric embedding, so "boundary" is
defined by index, following Laplacians.jl). For `n = 100000`:
- stride `⌈100000^{1/3}⌉ = 47`,
- **2128** vertices removed (indices `1, 48, 95, …`),
- **97872** interior vertices remain → that is the size of the SDDM solve.

### Relationship between the variants
For a given seed, **all four families derive from one base graph:**
- the boundary and non-boundary matrices are built from the **identical underlying
  graph** — the boundary version simply deletes the 2128 boundary nodes;
- the weighted and unweighted variants have the **identical graph topology** —
  *verified two ways*: (a) empirically, for seeds 1, 12, 35, 50 at `n=100000`,
  `uni_chimera(n,s)` and `wted_chimera(n,s)` have byte-identical sparsity patterns
  (same nnz, `colptr`, `rowval`); (b) structurally, in the `Laplacians.jl` code both
  routes seed identically (`hash(n, hash(s))`), build the graph from the **same**
  `semiwted_chimera_ijv(n)` and the same `thicken` draw, and diverge only at the final
  weighting (`unweight!` vs `rand_weight`), which cannot change the edge set.

This is why seed 35 induces the same numerical difficulty — and the same HyPre crash —
across `uni_chimera.s35`, `uni_bndry_chimera.s35`, `wted_chimera.s35`, and
`wted_bndry_chimera.s35`: they are all the same graph (re-weighted and/or with the
boundary removed).

## 4. Weighted vs. unweighted

- **`uni_*`** — all edge weights are **1** (`unweight!`). Conditioning comes purely
  from graph structure.
- **`wted_*`** — a **random weighting scheme** is applied (uniform random weights,
  and/or "smoothed" potential-based weights from a few diffusion steps
  `v ← A·D⁻¹v`). This produces **large edge-weight ratios**, which add ill-conditioning
  on top of the structural difficulty.

## 5. What "chimera" means (the underlying graph generator)

`chimera(n, s)` deterministically seeds the RNG with `hash(n, s)` and builds a graph
**recursively** (`Laplacians.jl/src/graphGenerators.jl`):

- **Base case** (small pieces / ~20% of the time): pick one *atomic* generator at
  random — `path`, `ring`/cycle, `complete binary tree`, `grown_graph`
  (preferential-attachment-like), a chunk of a `2-D grid`, `random 3-regular`
  (an expander), `Erdős–Rényi cluster`, or `random generalized ring` — then randomly
  permute the vertex labels.
- **Recursive case** (larger): split `n`, recursively build two smaller chimeras,
  and **combine** them by a randomly chosen operation — `join_graphs` (disjoint union
  + a few bridge edges), `product_graph` (graph Cartesian product), or
  `generalized_necklace` (chain copies of one graph around another). Each sub-piece is
  randomly rescaled and permuted.
- The result may be `thicken`-ed (overlay copies to raise density).

Because each seed produces a structurally different graph, the families span easy
(e.g. tree/grid draws — near-exact factorization, ~1 PCG iteration) to pathological
(e.g. long-path/cycle draws — condition number ~10⁹, where AMG breaks down).

## 6. Generation methodology

**Tool:** [`gen_chimera_sweep.jl`](gen_chimera_sweep.jl) (Julia + `Laplacians.jl`, `AMD`).

**Parameters used:**
- size `n = 100000`,
- seeds `s = 1, 2, …, 50` (50 instances per family),
- four families: `uni_chimera`, `wted_chimera`, `uni_bndry_chimera`, `wted_bndry_chimera`.

**Seeds (exact usage):**
- The **graph structure** is `uni_chimera(n, s)` / `wted_chimera(n, s)`; the seed `s`
  is the chimera index, hashed (`hash(n, s)`) to seed the RNG — so each `s` is fully
  deterministic and reproducible.
- The **right-hand side** is generated after `Random.seed!(rseed + s)` with the
  default `rseed = 0`, i.e. the RHS RNG seed is also `s`.

**Right-hand side:** `b = M · randn(size(M))`, then normalized `b ← b / ‖b‖` — the
SDDM2023 recipe. Because `b` lies in the column space of `M` by construction, it is a
valid RHS even for the singular (no-boundary) Laplacians.

**Orderings (the `CHIMERA_ORDER` env var) → output directory:**
- `none` → `chimera_ij/`     (input order),
- `nnz`  → `chimera_nnzsort/` (random tie-break shuffle then sort columns by ascending
  nnz, ≈ minimum-degree),
- `amd`  → `chimera_amd/`     (AMD fill-reducing order).

Incomplete-Cholesky preconditioner quality is ordering-sensitive, so the matrix is
reordered before the factorization; the reported head-to-head results use **AMD**.

**Output files per matrix** (`<base> = <family>.n100000.s<seed>`):
- `<base>.00000` + `<base>_rhs.00000` — HyPre IJ format (the **raw** matrix `M` and
  the RHS `b`). This is what BoomerAMG/HyPre solves.
- `<base>.mtx` — MatrixMarket, for the AC driver:
  - For the **`*_chimera`** (Laplacian) families this is the raw `L` (size `n`), solved
    in **graph mode** (`is_graph=1`).
  - For the **`*_bndry_chimera`** (SDDM) families this is the **augmented Laplacian**
    (size `n+1`): the SDDM `M` plus one appended "ground" node holding the row excess,
    so it is a true Laplacian again, solved in **physics mode** (`is_graph=0`). See the
    repository `README.md` for why this augmentation is required. The `_rhs.00000`
    still has the solved-system size (97872) entries.

**Exact commands used** (per family; replace the family name and ordering):
```bash
# input order
CHIMERA_ORDER=none julia gen_chimera_sweep.jl uni_chimera        100000 1 50 chimera_ij
CHIMERA_ORDER=none julia gen_chimera_sweep.jl wted_chimera       100000 1 50 chimera_ij
CHIMERA_ORDER=none julia gen_chimera_sweep.jl uni_bndry_chimera  100000 1 50 chimera_ij
CHIMERA_ORDER=none julia gen_chimera_sweep.jl wted_bndry_chimera 100000 1 50 chimera_ij
# AMD order (used for the reported results)
CHIMERA_ORDER=amd  julia gen_chimera_sweep.jl uni_chimera        100000 1 50 chimera_amd
CHIMERA_ORDER=amd  julia gen_chimera_sweep.jl wted_chimera       100000 1 50 chimera_amd
CHIMERA_ORDER=amd  julia gen_chimera_sweep.jl uni_bndry_chimera  100000 1 50 chimera_amd
CHIMERA_ORDER=amd  julia gen_chimera_sweep.jl wted_bndry_chimera 100000 1 50 chimera_amd
```

**Tolerances:** the comparison target is `1e-8` relative residual. HyPre is run at
`1e-8`. The AC driver is run at `1e-9` (`AC_TOL`) because it stops on MKL's *recursive*
residual, which sits a few × above the *true* `‖b−Ax‖`; the tighter target makes AC's
**true** residual reach `1e-8`.
