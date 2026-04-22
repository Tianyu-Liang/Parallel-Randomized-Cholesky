# Data Preparation for Solver Benchmarks

This directory contains scripts to download matrices and generate input files
for hypre (IJ format) and AMGX (MTX format).

## Matrices

15 test matrices split into two categories:

### Physics matrices (already have values, positive diagonal, negative off-diagonal)

| Matrix | Source | Notes |
|--------|--------|-------|
| parabolic_fem | SuiteSparse (Wissgott) | Laplacian (row sums = 0) |
| ecology1 | SuiteSparse (McRae) | True Laplacian (row sums = 0), treated as graph |
| ecology2 | SuiteSparse (McRae) | SDDM (4 rows have row sum = 1) |
| apache2 | SuiteSparse (GHS_psdef) | SDDM |
| G3_circuit | SuiteSparse (AMD) | SDDM |
| spe16m | Dropbox (SPE project) | Signs are flipped (negated during processing) |
| uniform_3D | Generated (Julia) | 3D uniform grid, ~100M nnz |
| aniso_contrast_3D | Generated (Julia) | Anisotropic contrast, ~100M nnz |
| poisson_contrast_3D | Generated (Julia) | Poisson with contrast coefficient, ~100M nnz |

### Graph matrices (adjacency -> Laplacian conversion)

| Matrix | Source | Format | Notes |
|--------|--------|--------|-------|
| belgium_osm | SuiteSparse (DIMACS10) | pattern symmetric | Road network |
| europe_osm | SuiteSparse (DIMACS10) | pattern symmetric | Road network |
| com-LiveJournal | SuiteSparse (SNAP) | pattern symmetric | Social network |
| delaunay_n24 | SuiteSparse (DIMACS10) | pattern symmetric | Delaunay triangulation |
| venturiLevel3 | SuiteSparse (DIMACS10) | pattern symmetric | Mesh |
| GAP-road | SuiteSparse (GAP) | integer symmetric | Road network (weighted edges) |

Graph Laplacian construction:
- Off-diagonal(i,j) = -|weight| (or -1 for pattern matrices)
- Diagonal(i) = sum of |off-diagonals in row i|
- This ensures positive diagonal, negative off-diagonal, and row sums = 0

## Step-by-step reproduction

All commands below assume you are in the `graph_sparsify/` directory.

### 1. Download matrices

```bash
cd data
bash download_matrices.sh
```

This downloads:
- Physics matrices from SuiteSparse to `physics/` (ecology1, ecology2, apache2, G3_circuit, parabolic_fem)
- SPE matrices from Dropbox to `physics/spe*/`
- Graph matrices should already be in `data/` (belgium_osm, europe_osm, etc.)

### 2. Generate Julia physics matrices

The 3 generated matrices (uniform_3D, aniso_contrast_3D, poisson_contrast_3D)
require Julia with the `Laplacians`, `SparseArrays`, `MatrixMarket`, `AMD` packages.

```bash
cd cpu_implementation
julia generate_raw_physics.jl
```

This creates:
- `physics/uniform_3D/uniform_3D.mtx`
- `physics/aniso_contrast_3D/aniso_contrast_3D.mtx`
- `physics/poisson_contrast_3D/poisson_contrast_3D.mtx`

Each is ~14.3M rows, ~100M nnz. Takes several minutes.

### 3. Generate solver input files

```bash
cd data
python3 generate_solver_inputs.py
```

This creates:
- `data/hypre/` — IJ format matrix (`.00000`) and RHS (`_rhs.00000`) files
- `data/amgx/` — MTX files with embedded RHS (`%%AMGX rhs` header)

Processing details:
- **Physics matrices**: used directly (positive diagonal, negative off-diagonal)
- **SPE matrices**: all values negated (original has negative diagonal, positive off-diagonal)
- **Graph matrices**: Laplacian constructed from adjacency (degree on diagonal, -weight off-diagonal)
- **RHS**: random uniform [0,1) then subtract mean to get zero-sum vector (MT19937, seed=0)

### 4. Run solvers

See the dedicated READMEs for running instructions and scripts:
- Hypre: [`hypre/README.md`](../hypre/README.md)
- AMGX: [`amgx/README.md`](../amgx/README.md)

## File format details

### Hypre IJ format

Matrix file (`name.00000`):
```
0 N-1          (row range)
0 N-1          (col range)
row col value  (0-indexed, one entry per line, both triangles)
...
```

RHS file (`name_rhs.00000`):
```
0 N-1          (row range)
row value      (0-indexed)
...
```

### AMGX MTX format

```
%%MatrixMarket matrix coordinate real symmetric
%%AMGX rhs
N N nnz_lower_triangle
row col value  (1-indexed, lower triangle only)
...
N              (RHS length)
value          (one per line)
...
```

## Notes

- The RHS uses numpy's MT19937 (seed=0), which differs from C++ std::mt19937.
  The exact values differ but the statistical property (uniform, zero-mean) is the same.
- ecology1 is a true Laplacian (all row sums = 0) and does not require the SDDM
  augmentation that other physics matrices have. It is essentially a graph Laplacian
  from an ecological connectivity model, so it is treated as a graph problem rather
  than a physics problem when benchmarking.
- ecology2 has 4 boundary rows with row sum = 1 (SDDM, not pure Laplacian).
- SPE matrices are SDDM (row sums significantly negative), not Laplacian.
