# ParAC benchmark implementations

## Overview

- `cpu_implementation/` — CPU factorization only (ParAC).
- `experiment/` — Complete CPU pipeline (factorization + PCG solve). Also contains `independent_cg` for solving with a pre-computed preconditioner. Requires MKL.
- `gpu_implementation/` — GPU factorization + solve. Has separate drivers for graph Laplacians (`driver.cu`) and physics/SDDM matrices (`driver_physics.cu`).
- `data/` — Scripts and instructions for the original benchmark matrices, the randomized Chimera robustness study, and baseline-solver inputs. See [`data/README.md`](data/README.md).
- `hypre/` and `amgx/` — Run scripts and READMEs for the baseline solvers (BoomerAMG and AMGX).

## Prerequisites

1. Download [fast_matrix_market](https://github.com/alugowski/fast_matrix_market) and note its `include/` directory.
2. Run `bash data/download_matrices.sh` from the repository root. For example, `parabolic_fem` is placed at `physics/parabolic_fem/parabolic_fem.mtx`.
3. Use `cpu_implementation/write_graph.jl` to produce reordered matrices. See the `cpu_implementation/produce_*.jl` files for examples.

**Make sure to create a folder for each matrix. Folders must be manually created for matrices that are not from SuiteSparse (e.g., 3D uniform Poisson).**

Generated matrices, solver logs, compiled binaries, and local paper drafts are excluded by `.gitignore` and should not be committed.

## Build

Pass the `fast_matrix_market` include directory to each build. The compiler, CUDA compiler,
and GPU architecture can be overridden with `CXX`, `NVCC`, and `GPU_ARCH` respectively.
If MKL or the CUDA math libraries are not already on the compiler's search path, also set
`MKL_INCLUDE`/`MKL_LIB_DIR` or `CUDA_LIB_DIR`.

```bash
export FMM_INCLUDE=/path/to/fast_matrix_market/include
export MKLROOT=/path/to/oneapi/mkl/latest
make -C cpu_implementation FMM_INCLUDE="$FMM_INCLUDE"
make -C experiment FMM_INCLUDE="$FMM_INCLUDE" \
  MKL_INCLUDE="$MKLROOT/include" MKL_LIB_DIR="$MKLROOT/lib/intel64"
make -C gpu_implementation FMM_INCLUDE="$FMM_INCLUDE" GPU_ARCH=sm_80 \
  NVCC=/path/to/cuda/bin/nvcc \
  CUDA_LIB_DIR=/path/to/cuda/math_libs/lib
```

The resulting executables are:

- `cpu_implementation/driver` — CPU factorization.
- `experiment/driver` — CPU factorization followed by PCG.
- `experiment/independent_cg` — PCG with a precomputed factor.
- `gpu_implementation/driver` — GPU graph-Laplacian driver.
- `gpu_implementation/driver_physics` — GPU physics/SDDM driver.

## CPU factorization (`cpu_implementation/driver`)

```bash
./driver path/to/matrix-amd.mtx 32 "" <is_graph>
```

Arguments:
1. Matrix file path (.mtx)
2. Number of threads
3. Output path for the computed factorization (empty string `""` = don't write)
4. `is_graph` — **`1` = graph Laplacian** (factor as-is); **`0` = physics/SDDM** (the matrix is an augmented Laplacian; the appended last row/column is trimmed first). Same `1`/`0` convention as `experiment/driver` and `independent_cg`.

```bash
./driver path/to/matrix-amd.mtx 32 "" 1     # graph Laplacian
./driver path/to/matrix-amd.mtx 32 "" 0     # physics / SDDM
```

See `experiment/graph_test_amd.sh` (uses `1`) and `experiment/physics_test_amd.sh` / `experiment/spe_test_amd.sh` (use `0`).

> ⚠️ **SDDM INPUT MUST BE AUGMENTED — `is_graph` ALONE IS NOT ENOUGH. (Easy to get wrong.)**
> The driver factorizes a **graph Laplacian** (zero row sums). It does **NOT** auto-convert an SDDM
> matrix. The two modes are:
> - **`1` (graph):** input is a true Laplacian — diagonal exactly equals the sum of off-diagonals, so
>   every row sums to 0.
> - **`0` (physics/SDDM):** input must ALREADY be an **augmented Laplacian** — your SDDM matrix `M`
>   (size *k*) with **one extra "ground" node appended as the LAST row/column** (index *k+1*) holding
>   each row's excess, so the augmented matrix is again a true Laplacian. Physics mode only **trims**
>   that last row/column after factorizing; **it does not append it for you.**
>
> **The trap:** if you feed a *raw, un-augmented* SDDM matrix (nonzero row sums) you get a **silent,
> wrong result, not an error** — graph mode mishandles the excess diagonal and the preconditioner is
> badly degraded (e.g. a chimera SDDM that should solve in ~20 iters instead stalls at 1000); physics
> mode would trim a real interior node and solve a *different* system. **Neither mode is correct for a
> raw SDDM.**
>
> **How to augment** a raw SDDM `M`: let `e = M·1` (row sums; `eᵢ > 0` exactly on rows that lost an
> edge to a removed boundary node). Append a ground node as the last index:
> `L = [ M  -e ; -eᵀ  Σe ]` — now `L` is a true Laplacian. Factor `L` (`L ≈ RᵀR`) and drop the ground
> node's row/col from the factor → `G` with **`GᵀG ≈ M`** (exact for exact Cholesky, since the ground
> node is eliminated last). That trim is exactly what `is_graph=0` does. The ground node must stay the
> **last** index even after any fill-reducing reorder (reorder the interior, then append).
> `data/gen_chimera_sweep.jl` does this automatically for the `*_bndry_chimera` (SDDM) families: it
> writes the **augmented** Laplacian to `.mtx` (run with `is_graph=0`) while keeping the raw `M` in the
> HyPre `.00000` files. `*_chimera` (true Laplacian) families write the raw matrix (run with `is_graph=1`).

## CPU factorization + solve (`experiment/driver`)

Same as the factorization driver, plus it runs the PCG solve and takes two optional trailing args:

```bash
./driver path/to/matrix.mtx 32 "" <is_graph> [tol] [rhsfile]
```

5. (Optional) `tol` — relative tolerance for the PCG solve (default `1e-7`).
6. (Optional) `rhsfile` — read the right-hand side `b` from this file (HyPre IJ vector format: a header line, then `idx val` per line). Use this to solve the *same* `b` another solver uses.

> ⚠️ **RHS LIMITATION — THE DEFAULT RHS DOES NOT WORK FOR DISCONNECTED GRAPHS.**
> With no `rhsfile`, the driver builds a **zero-sum random** RHS, which is only in the image of `A`
> for a **CONNECTED** graph. For a **DISCONNECTED** graph each component has its own constant null
> vector, so a globally-zero-sum vector is generally **NOT in the problem span** and the solve will
> not converge. For disconnected inputs you must either **(a) supply `rhsfile` with `b = A·x`**
> (always in the image), **or (b) detect the connected components and build a RHS that is zero-sum
> within each component.**

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

Those scripts expect precomputed factor files named `_ic_amd.mtx`. Generate them by passing that path as argument 3 to `cpu_implementation/driver` before running `independent_cg`.

## GPU (`gpu_implementation/`)

There are two drivers:
- `driver.cu` — for graph Laplacians
- `driver_physics.cu` — for physics/SDDM matrices

`make all` builds both binaries (`driver` and `driver_physics`). They have the same interface:
```bash
./driver path/to/graph-nnz-sorted.mtx 512 1 7e-7 [rhsfile] [pool_mult]
./driver_physics path/to/sddm-nnz-sorted.mtx 512 1 7e-7 [rhsfile] [pool_mult]
```

Arguments:
1. Matrix file path (.mtx)
2. Number of thread blocks
3. Whether to run the solve phase (`0` = skip, `1` = run)
4. Target relative tolerance (scientific notation)
5. (Optional) `rhsfile` — read the right-hand side `b` from this file (HyPre IJ vector format: a
   header line, then `idx val` per line), so the GPU solves the *same* `b` another solver uses. Omit
   it, or pass `""`, to use the default zero-sum random RHS (valid only for **connected** graphs).
6. (Optional) `pool_mult` — the factorization **edge-pool capacity as a multiple of nnz** (default
   `4`, i.e. the pool holds `4 × nnz` entries). Increase it for **dense** matrices whose factor
   overflows the default `4×` pool — the symptom is an **illegal-memory crash or a hang** (the pool
   has no overflow guard). E.g. `./driver dense.mtx 512 1 7e-7 "" 8` uses an `8×` pool with no
   rhsfile. The CPU driver grows storage dynamically and needs no such flag.

> ⚠️ **Dense-matrix note:** the GPU factorization pre-allocates a fixed `pool_mult × nnz` edge pool.
> If a matrix's factor exceeds it, the driver either crashes (illegal memory access) or deadlocks,
> producing no output. Re-run that matrix with a larger `pool_mult` (argument 6).

> **com-LiveJournal (special case).** It is the **densest matrix in the benchmark**, and the regime
> where classical AMG (HyPre/AMGX) runs **out of memory** while ParAC does not — hence the large
> speedup reported for it. It needs a **non-default launch**: **96 blocks** (argument 2) and **8 warps
> per squad/block**. The warp count is a **compile-time** constant, so set `WARPS_PER_SQUAD` and
> `HOST_WARPS_PER_SQUAD` to `8` in `gpu_implementation/auxilliary.hpp` and **recompile** (it is not a
> runtime argument). It also wants a **pool multiplier (argument 6) ≥ 6** depending on ordering. Run it
> on its own (it is commented out of `test_script_graph` for this reason), e.g. after recompiling
> with 8 warps:
> ```bash
> ./driver ../data/com-LiveJournal/com-LiveJournal-nnz-sorted.mtx 96 1 1e-7 "" 6
> ```

See `gpu_implementation/test_script_graph` and `gpu_implementation/test_script_physics_nnz_sort` for examples. Run them from `gpu_implementation/` after building both drivers.

## Randomized Chimera robustness experiment

The randomized experiment compares ParAC with BoomerAMG and AMGX on four Chimera matrix families, with 50 seeds per family (200 systems per comparison).

- Matrix definitions and generation: [`data/CHIMERA_MATRICES.md`](data/CHIMERA_MATRICES.md)
- CPU ParAC versus BoomerAMG: `data/run_ac_vs_hypre.sh`
- GPU ParAC versus AMGX: [`data/amgx_vs_gpu_procedure.md`](data/amgx_vs_gpu_procedure.md) and `data/run_amgx_vs_gpu.sh`
- CPU summary: `data/chimera_amd/SUMMARY_ac_vs_hypre_tol1e-8.txt`
- GPU summary: `data/chimera_nnzsort/SUMMARY_amgx_vs_gpu_tol1e-8.txt`

The large per-seed matrices and logs are generated locally and ignored; only the reproduction scripts, configuration, documentation, and compact summary tables belong in Git.

The runner scripts derive repository paths automatically. Set `IJ`, `AMGX`, and optionally `AMGX_LIB` to the corresponding external installations when running the baseline solvers.
