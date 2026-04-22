# Building Hypre on Perlmutter (NERSC)

## Prerequisites

Make sure you are using the GNU programming environment (default on Perlmutter):

```bash
module load PrgEnv-gnu
```

## Clone

```bash
cd /pscratch/sd/t/tianyul/hypre
git clone https://github.com/hypre-space/hypre.git hypre_newest
cd hypre_newest/src
```

## Configure (CPU with MPI + OpenMP, no CUDA)

```bash
./configure \
  --prefix=/pscratch/sd/t/tianyul/hypre/hypre_newest/install \
  --with-MPI \
  --with-openmp \
  --without-cuda \
  CC=cc CXX=CC FC=ftn \
  CFLAGS="-O2 -fopenmp" CXXFLAGS="-O2 -fopenmp" FCFLAGS="-O2 -fopenmp" LDFLAGS="-fopenmp"
```

Key flags:
- `CC=cc CXX=CC FC=ftn` -- Cray compiler wrappers (handle MPI automatically)
- `CFLAGS="-O2 -fopenmp"` -- required because the auto-detected OpenMP flag (`-qsmp=omp`) is wrong for GNU compilers behind the Cray wrappers. Must include `-O2` since setting CFLAGS overrides the default optimization level
- `LDFLAGS="-fopenmp"` -- ensures OpenMP is linked (default would have `-qsmp=omp`)
- `--without-cuda` -- must be explicit, otherwise configure tries to find CUDA and fails

## Build and Install

```bash
make -j16
make install
```

The library is installed to:
- `/pscratch/sd/t/tianyul/hypre/hypre_newest/install/lib/`
- `/pscratch/sd/t/tianyul/hypre/hypre_newest/install/include/`

## Building the test drivers

```bash
cd /pscratch/sd/t/tianyul/hypre/hypre_newest/src/test
make ij CC=cc CXX=CC COPTS="-fopenmp" LINKOPTS="-fopenmp" LIBS="-L../hypre/lib -lHYPRE -lm -fopenmp"
```

Notes:
- The examples/test Makefiles don't inherit configure settings, so compiler and OpenMP flags must be passed manually
- `LIBS` must include the original `-L../hypre/lib -lHYPRE -lm` plus `-fopenmp` (overriding replaces the default entirely)

## Running benchmarks

The benchmark script is at `script_run.sh`. It runs all 15 matrices with AMG-PCG (`-solver 1 -tol 1e-6`).

```bash
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
bash script_run.sh
```

The data files are in `/pscratch/sd/t/tianyul/randla/graph_sparsify/data/hypre/`.
See `data/README.md` for how to regenerate them.

For multi-thread scaling, use `hypre_omp_job.sh` which sweeps over thread counts (1, 2, 4, 8, 16, 32).

## Running (general)

### Single process, multi-threaded

```bash
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
srun -n 1 -c 32 --cpu-bind=cores ./ij -fromfile <matrix.ij> -rhsfromfile <rhs.ij> -solver 1 -tol 1e-6
```

### Multi-process

```bash
srun -n 4 ./ij -fromMMfile <matrix.mtx> -solver 1 -tol 1e-6
```

### Solver IDs

| ID | Solver |
|----|--------|
| 0  | AMG (standalone, default) |
| 1  | AMG-PCG |
| 2  | DS-PCG |
| 3  | AMG-GMRES |
| 4  | DS-GMRES |
| 61 | AMG-FlexGMRES |

## Input file formats

### IJ format (for `-fromfile`)

Hypre appends the MPI rank to the filename (e.g., `matrix.ij.00000` for rank 0). For single-process runs, rename/copy the file:

```bash
cp matrix.ij matrix.ij.00000
cp rhs.ij rhs.ij.00000
```

File format:
```
first_row last_row       # e.g., 0 999
first_col last_col       # e.g., 0 999
row col value            # one entry per line, 0-indexed
row col value
...
```

RHS file format:
```
first_row last_row
row value
row value
...
```

### MatrixMarket format (for `-fromMMfile`)

Reads standard `.mtx` files directly. Supports `real symmetric coordinate` format. Handles symmetric expansion internally. No rank suffix needed. Does not support a separate RHS file (defaults to b=[1,...,1]).

## Known issues

- `test_ij` has compilation errors in `hypre_set_precond.c` (missing function declarations). Use `ij` instead.
- The Cray MPI wrappers hide include/library paths from non-wrapper compilers (e.g., nvcc). If building with CUDA, you may need to explicitly pass `$CRAY_MPICH_DIR/include` and `$CRAY_MPICH_DIR/lib`.
