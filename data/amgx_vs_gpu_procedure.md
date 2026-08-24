# AMGX vs GPU-ParAC sweep — run procedure

Head-to-head of **AMGX** (classical AMG-preconditioned PCG) vs **GPU ParAC**, on the **same**
chimera matrices and the **same RHS** `b = M·g`. GPU ParAC uses the **nnz-sort** ordering.
Driver: [`run_amgx_vs_gpu.sh`](run_amgx_vs_gpu.sh). Matrices: see [`CHIMERA_MATRICES.md`](CHIMERA_MATRICES.md).

**Run everything on a GPU compute node (`salloc`), never on a login node.**

---

## Prerequisites (one-time)

Run the commands in this document from the repository's `data/` directory:

```bash
cd /path/to/graph_sparsify_test/data
```

1. **Matrices generated** in `chimera_nnzsort/` — all four families × 50 seeds
   (`uni_chimera`, `wted_chimera`, `uni_bndry_chimera`, `wted_bndry_chimera`). If missing:
   ```bash
   for fam in uni_chimera wted_chimera uni_bndry_chimera wted_bndry_chimera; do
     CHIMERA_ORDER=nnz julia gen_chimera_sweep.jl $fam 100000 1 50 chimera_nnzsort
   done
   ```
2. **GPU drivers built** with the current interface (rhsfile arg `argv[5]`, pool-multiplier arg
   `argv[6]`, and `MAX_ITERS=1000` in `solver.hpp`):
   ```bash
   cd ../gpu_implementation && make      # builds both driver + driver_physics (first target = all)
   ```
   Rebuild whenever you change `driver.cu`, `driver_physics.cu`, `solver.hpp`, or `auxilliary.hpp`.

---

## The procedure

### 1. Full pass at the default pool (4)
```bash
AMGX=/path/to/amgx_capi AMGX_LIB=/path/to/amgx/build \
  ./run_amgx_vs_gpu.sh chimera_nnzsort
```
Runs all 200 systems unattended. Output: `chimera_nnzsort/SUMMARY_amgx_vs_gpu_tol1e-8.txt`
with columns `matrix | AMGX(it/relres/verd) | GPU-AC(it/relres/verd) | amgx_opc | pool`.
Per-seed detail in `<base>.amgx.log` and `<base>.gpuac.log` (the latter prints `edge pool: Nx nnz`).

The sweep is **resumable**: each seed writes `<base>.row`, and the SUMMARY is collated (sorted) from
all `.row` files at the end. Re-running any subset updates just those rows and re-collates the table.

### 2. Fix the two GPU-AC failure modes

There are exactly two, and they are separable:

| symptom in GPU-AC column | cause | fix |
|---|---|---|
| `300/…/WEAK` (or `1000/…/WEAK`) | hit the iteration cap, residual still descending | already raised `MAX_ITERS` 300→1000; just re-run at pool 4 |
| `HANG(>60s)` or `-/-/FAIL` | factorization **pool overflow** on a dense seed (crash or deadlock) | re-run at a larger `GPU_POOL` (8, then 12/16) |

Extract both buckets from the current summary **first** (re-running changes it), then fix each:
```bash
S=chimera_nnzsort/SUMMARY_amgx_vs_gpu_tol1e-8.txt
mapfile -t WEAK     < <(awk -F'|' '$3 ~ /WEAK/      {gsub(/ /,"",$1); print "chimera_nnzsort/"$1}' "$S")
mapfile -t OVERFLOW < <(awk -F'|' '$3 ~ /HANG|FAIL/ {gsub(/ /,"",$1); print "chimera_nnzsort/"$1}' "$S")
echo "WEAK (${#WEAK[@]}):";     printf '  %s\n' "${WEAK[@]}"
echo "OVERFLOW (${#OVERFLOW[@]}):"; printf '  %s\n' "${OVERFLOW[@]}"

# iter-capped ones: pool 4 is fine, they just need the higher MAX_ITERS (so rebuild first)
./run_amgx_vs_gpu.sh "${WEAK[@]}"

# pool-overflow ones: bigger pool
GPU_POOL=8 ./run_amgx_vs_gpu.sh "${OVERFLOW[@]}"
# if any STILL hang/crash at 8 (the densest, op-cx ~20+), bump just those:
# GPU_POOL=12 ./run_amgx_vs_gpu.sh chimera_nnzsort/<base> ...
```

### 3. Read the final table
```bash
cat chimera_nnzsort/SUMMARY_amgx_vs_gpu_tol1e-8.txt
```
The `pool` column shows the capacity multiplier each row used (mostly 4, a few 8/12 on dense seeds).

---

## Knobs (env vars on `run_amgx_vs_gpu.sh`)

| var | default | meaning |
|---|---|---|
| `GPU_POOL` | 4 | factorization edge-pool multiplier (`pool × full_nnz`); raise for dense seeds |
| `GPU_TIMEOUT` | 60 | seconds before a hung GPU run is killed (→ `HANG`); matrices are small so 60 is ample |
| `GPU_BLOCKS` | 1024 | GPU thread blocks (`argv[2]`) |
| `GPU_TOL` | 1e-8 | GPU stop target (reported value is the **true** residual `‖b−Ax‖/‖b‖`) |
| `TOL` | 1e-8 | comparison target the verdicts (PASS/WEAK/FAIL) are graded against |
| `AMGX` | `amgx_capi` from `PATH` | AMGX executable |
| `AMGX_LIB` | empty | AMGX build/library directory prepended to `LD_LIBRARY_PATH` |

`MAX_ITERS` (1000) is **compile-time** in `gpu_implementation/solver.hpp` — change + rebuild to alter it.

---

## Consistency notes (so the numbers are comparable)

- **nnz is the FULL count** (both triangles, diagonal once) for both solvers. The `.mtx` is stored
  lower-triangular but the GPU reader expands it on load (`nonZeros() == 2·stored − n`), matching the
  `general` `.amgx` AMGX reads. So `pool × nnz` and AMGX's operator complexity share the same baseline.
- **Same `b = M·g`**: AMGX reads it from the `.amgx`; the GPU driver reads the matching
  `<base>_rhs.00000` via `argv[5]`.
- **Family routing**: `*_chimera` (Laplacian) → `driver` (graph) on the raw `.mtx`; `*_bndry` (SDDM)
  → `driver_physics` on the **augmented** `.mtx` (the script picks automatically).
- This sweep is GPU-AC (nnz-sort) vs AMGX. The **CPU-AC vs HyPre** comparison (AMD ordering) is the
  separate `run_ac_vs_hypre.sh chimera_amd`. **com-LiveJournal** is its own special run (96 blocks,
  8 warps-per-squad via recompile, `pool ≥ 6`) — see the GPU section of the top-level `README.md`.
