# Instructions

## Overview

- `cpu_implementation/` — CPU factorization only (ParAC).
- `experiment/` — Complete CPU pipeline (factorization + PCG solve). Also contains `independent_cg` for solving with a pre-computed preconditioner. Requires MKL.
- `gpu_implementation/` — GPU factorization + solve. Has separate drivers for graph Laplacians (`driver.cu`) and physics/SDDM matrices (`driver_physics.cu`).
- `data/` — Scripts and instructions for downloading/generating the 15 benchmark matrices. See `data/README.md`.
- `hypre/` and `amgx/` — Run scripts and READMEs for the baseline solvers (BoomerAMG and AMGX).

## Prerequisites

1. Download [fast_matrix_market](https://github.com/alugowski/fast_matrix_market) and update the include path in the makefiles.
2. Download the relevant matrices from SuiteSparse and place them into folders. For instance, `parabolic_fem` should be at `data/parabolic_fem/parabolic_fem.mtx`.
3. Use `write_graph.jl` from `cpu_implementation/` to produce reordered matrices. See the `produce_*.jl` files for examples.

**Make sure to create a folder for each matrix. Folders must be manually created for matrices that are not from SuiteSparse (e.g., 3D uniform Poisson).**

## CPU factorization (`cpu_implementation/driver`)

For physics/SDDM matrices (not originally a Laplacian — appends a row/column to make it one):
```bash
./driver path/to/matrix-amd.mtx 32 "" 1
```

For graph Laplacians (already a Laplacian, no augmentation needed):
```bash
./driver path/to/matrix-amd.mtx 32 ""
```

Arguments:
1. Matrix file path (.mtx)
2. Number of threads
3. Output path for the computed factorization (empty string `""` = don't write)
4. (Optional) Any value here triggers physics mode (the matrix will be augmented and trimmed). Omit for graph Laplacians.

See `experiment/physics_test_amd.sh`, `experiment/graph_test_amd.sh`, and `experiment/spe_test_amd.sh` for examples.

## CPU factorization + solve (`experiment/driver`)

Same interface as the CPU factorization driver above. The difference is that this version also runs the PCG solve after factorization.

**Requires MKL. On Perlmutter, use `module load intel` to set up the paths.**

## Independent CG solver (`experiment/independent_cg`)

Solves with a pre-computed incomplete Cholesky preconditioner using MKL's CG:
```bash
./independent_cg path/to/matrix.mtx path/to/preconditioner.mtx is_graph [max_iter] [rel_tol]
```

Arguments:
1. Matrix file path (.mtx)
2. Preconditioner factor file path (.mtx)
3. `1` for graph Laplacian, `0` for physics/SDDM (physics mode removes the last row/column)
4. (Optional) Maximum iterations (default: 1000)
5. (Optional) Relative tolerance (default: 1e-7)

Examples:
```bash
# Graph problem with custom max_iter and tolerance
./independent_cg "../data/europe_osm/europe_osm-amd.mtx" "../data/europe_osm/_ic_amd.mtx" 1 10000 5e-7

# Physics problem with defaults
./independent_cg "../physics/parabolic_fem/parabolic_fem-amd.mtx" "../physics/parabolic_fem/_ic_amd.mtx" 0
```

See `experiment/ichol_graph.sh` and `experiment/ichol_physics.sh` for examples.

## GPU (`gpu_implementation/`)

There are two drivers:
- `driver.cu` — for graph Laplacians
- `driver_physics.cu` — for physics/SDDM matrices

Switch which one to compile in the makefile. Both have the same interface:
```bash
./driver path/to/matrix-nnz-sorted.mtx 512 1 7e-7
```

Arguments:
1. Matrix file path (.mtx)
2. Number of thread blocks
3. Whether to run the solve phase (`0` = skip, `1` = run)
4. Target relative tolerance (scientific notation)

See `gpu_implementation/physics_test_nnz_sort.sh` for examples.
