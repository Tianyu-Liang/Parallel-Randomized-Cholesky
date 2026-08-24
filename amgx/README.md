# Running AMGX Benchmarks on Perlmutter

## Prerequisites

Requires a GPU node (A100 80GB). Request an interactive session:

```bash
salloc -N 1 -t 30 -C gpu -q interactive -A <account> --gpus=1
```

Build AMGX separately and record its source/build location, for example:

```bash
export AMGX_ROOT=/path/to/AMGX
```

## Running benchmarks

From the ParAC repository root, `amgx/run_script` runs the original benchmark suite.
Provide the AMGX executable and the two AMGX configuration files through environment variables:

```bash
AMGX="$AMGX_ROOT/build/examples/amgx_capi" \
PCG_V="$AMGX_ROOT/src/configs/PCG_V.json" \
AMG_CG="$AMGX_ROOT/src/configs/AMG_CLASSICAL_CG.json" \
bash amgx/run_script
```

This runs all 15 matrices with two configs:
- **PCG_V.json**: PCG with V-cycle AMG preconditioner (D2 interpolation, Block Jacobi smoother)
- **AMG_CLASSICAL_CG.json**: Standalone AMG with CG cycle (more robust for difficult graphs)

Config assignment:

| Config | Matrices |
|--------|----------|
| PCG_V | parabolic_fem, ecology2, apache2, G3_circuit, spe16m, uniform_3D, aniso_contrast_3D, poisson_contrast_3D, venturiLevel3, com-LiveJournal |
| AMG_CLASSICAL_CG | GAP-road, europe_osm, delaunay_n24, belgium_osm, ecology1 |

Both configs use tolerance 1e-6, max 200 iterations.

## Data files

Matrix files (MTX with embedded RHS) are in:
```
data/amgx/
```

The runner derives this path from the repository automatically; override it with `DATA` if
needed. See [`data/README.md`](../data/README.md) for how to regenerate the inputs.

## Running a single matrix

```bash
"$AMGX_ROOT/build/examples/amgx_capi" \
  -m <path-to-matrix>.mtx \
  -c "$AMGX_ROOT/src/configs/PCG_V.json"
```

## Randomized Chimera comparison

The Chimera experiment uses the repository configuration `data/amgx_pcg_amg_t1e8.json`.
Run the GPU comparison from `data/`:

```bash
cd data
AMGX="$AMGX_ROOT/build/examples/amgx_capi" \
AMGX_LIB="$AMGX_ROOT/build" \
./run_amgx_vs_gpu.sh chimera_nnzsort
```

The script derives all ParAC and converter paths from the repository. See
[`data/amgx_vs_gpu_procedure.md`](../data/amgx_vs_gpu_procedure.md) for generation,
rerun, and summary instructions.
