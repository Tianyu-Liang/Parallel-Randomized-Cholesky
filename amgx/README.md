# Running AMGX Benchmarks on Perlmutter

## Prerequisites

Requires a GPU node (A100 80GB). Request an interactive session:

```bash
salloc -N 1 -t 30 -C gpu -q interactive -A <account> --gpus=1
```

AMGX build is at `/pscratch/sd/t/tianyul/amgx/amgnew/AMGX/build/`.

## Running benchmarks

The benchmark script is at `script_run.sh`. Run from the AMGX `build/` directory:

```bash
cd /pscratch/sd/t/tianyul/amgx/amgnew/AMGX/build
bash /pscratch/sd/t/tianyul/randla/graph_sparsify_fresh/amgx/script_run.sh
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
/pscratch/sd/t/tianyul/randla/graph_sparsify/data/amgx/
```

See `data/README.md` for how to regenerate them.

## Running a single matrix

```bash
cd /pscratch/sd/t/tianyul/amgx/amgnew/AMGX/build
examples/amgx_capi -m <path-to-matrix>.mtx -c ../src/configs/PCG_V.json
```


