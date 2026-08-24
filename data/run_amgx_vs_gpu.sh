#!/usr/bin/env bash
# Head-to-head: AMGX (classical AMG-preconditioned PCG) vs GPU ParAC, on the SAME chimera
# matrices and the SAME RHS b = M*g. Both run on the GPU.
#
# Per the chosen setup: GPU ParAC uses nnz-sort ordering (pass a chimera_nnzsort dir/base).
#   - *_chimera   (Laplacian): GPU graph driver on the raw .mtx (size n).
#   - *_bndry     (SDDM):      GPU physics driver on the AUGMENTED .mtx (size n+1, ground trimmed).
#   AMGX solves the raw matrix from the HyPre IJ (.amgx, auto-converted), identical (matrix, RHS).
#
# RUN inside a GPU allocation (both solvers are GPU; GPU driver is run DIRECTLY, no srun):
#   salloc -N 1 -C gpu -q interactive -t 60:00 -A <acct>
#   ./run_amgx_vs_gpu.sh chimera_nnzsort                      # all 200 systems
#   ./run_amgx_vs_gpu.sh chimera_nnzsort/uni_chimera.n100000.s12   # one base
#
# Env: TOL GPU_TOL GPU_BLOCKS MODE  AMGX CFG AMGX_LIB  GPU_GRAPH GPU_PHYS
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
GPU_GRAPH=${GPU_GRAPH:-$ROOT/gpu_implementation/driver}
GPU_PHYS=${GPU_PHYS:-$ROOT/gpu_implementation/driver_physics}
AMGX=${AMGX:-amgx_capi}
CFG=${CFG:-$ROOT/data/amgx_pcg_amg_t1e8.json}
AMGX_LIB=${AMGX_LIB:-}
CONV=$ROOT/data/gen_amgx_from_ij.py
MODE=${MODE:-dDDI}
TOL=${TOL:-1e-8}                       # comparison target (verdicts vs this)
GPU_TOL=${GPU_TOL:-1e-8}               # GPU stop target; report is the TRUE residual (normalized diff)
GPU_BLOCKS=${GPU_BLOCKS:-1024}
[ -z "$AMGX_LIB" ] || export LD_LIBRARY_PATH="$AMGX_LIB:${LD_LIBRARY_PATH:-}"

# verdict vs tol; 10x allowance for monitor/recursive-vs-true wiggle
classify() { awk -v r="$1" -v t="$TOL" 'BEGIN{
  if(r=="-"){print "FAIL";exit}
  f=r/t; print (f<=1e1)?"PASS":(r<=1e-2)?"WEAK":"FAIL"}'; }

declare -a BASE=()
for arg in "$@"; do
  if [ -d "$arg" ]; then
    while IFS= read -r f; do BASE+=("${f%.mtx}"); done < <(find "$arg" -name '*.mtx' | sort)
  else
    BASE+=("${arg%.mtx}")
  fi
done
[ "${#BASE[@]}" -eq 0 ] && { echo "no .mtx systems found"; exit 1; }

OUTDIR=$(dirname "${BASE[0]}")
SUMMARY="$OUTDIR/SUMMARY_amgx_vs_gpu_tol${TOL}.txt"
# RESUMABLE design: each seed's result is written to <base>.row, and the SUMMARY is COLLATED from all
# <base>.row files at the end (sorted). So the workflow is:
#   1) run the whole set at GPU_POOL=4 (default); dense seeds that overflow show FAIL/HANG.
#   2) re-run JUST those seeds at a higher pool, e.g.  GPU_POOL=8 ./run_amgx_vs_gpu.sh <those bases>
#      -- their .row files update and the SUMMARY is re-collated with the gaps filled.
# The "pool" column records the GPU_POOL each row was produced with (so you see "mostly 4, a few 8").
HDR=$(printf "%-42s | %-24s | %-24s | %8s | %4s" "matrix" "AMGX(it/relres/verd)" "GPU-AC(it/relres/verd)" "amgx_opc" "pool")
echo "$HDR"; printf '%.0s-' {1..114}; echo

for base in "${BASE[@]}"; do
  name=$(basename "$base"); rhs="${base}_rhs.00000"

  # ---- AMGX (raw matrix via .amgx; auto-convert from the IJ files) ----
  [ -f "$base.amgx" ] || python3 "$CONV" "$base" >/dev/null 2>&1
  xlog="$base.amgx.log"
  $AMGX -mode "$MODE" -m "$base.amgx" -c "$CFG" > "$xlog" 2>&1
  xit=$(grep -oP 'Total Iterations:\s*\K[0-9]+' "$xlog" | tail -1)
  xr=$(grep -oP 'Final Residual:\s*\K[0-9.eE+-]+' "$xlog" | tail -1)
  xoc=$(grep -oP 'Operator Complexity:\s*\K[0-9.eE+-]+' "$xlog" | tail -1)
  xcell=$(printf "%s/%s/%s" "${xit:--}" "${xr:--}" "$(classify "${xr:--}")")

  # ---- GPU ParAC (graph vs physics by family; same b=M*g via rhsfile) ----
  case "$name" in *bndry*) GPU=$GPU_PHYS;; *) GPU=$GPU_GRAPH;; esac
  glog="$base.gpuac.log"
  # GPU_TIMEOUT guards the intermittent factorization deadlock on dense seeds: if the driver hangs,
  # kill it after the timeout and keep the sweep moving (the seed is marked HANG instead of blocking).
  # GPU_POOL = factorization edge-pool multiplier (argv[6], default 4). Raise it (e.g. 8) so the
  # dense seeds don't overflow the pool and crash/hang. The driver's "edge pool: Nx nnz" line (in the
  # .gpuac.log) records it per seed too; nnz there is the FULL (both-triangle) count, same as AMGX.
  timeout "${GPU_TIMEOUT:-60}" "$GPU" "$base.mtx" "$GPU_BLOCKS" 1 "$GPU_TOL" "$rhs" "${GPU_POOL:-4}" > "$glog" 2>&1
  gstat=$?
  git=$(grep -oP 'final iteration:\s*\K[0-9]+' "$glog" | tail -1)
  gr=$(grep -oP 'normalized diff norm:\s*\K[0-9.eE+-]+' "$glog" | tail -1)
  [ -n "$gr" ] && gr=$(awk -v v="$gr" 'BEGIN{printf "%.6e", v}')   # %.16f from driver -> scientific
  if [ "$gstat" = 124 ]; then
    gcell="HANG(>${GPU_TIMEOUT:-60}s)/-/-"
  else
    gcell=$(printf "%s/%s/%s" "${git:--}" "${gr:--}" "$(classify "${gr:--}")")
  fi

  # write this seed's row (incl. the pool multiplier used) to <base>.row and echo it live
  row=$(printf "%-42s | %-24s | %-24s | %8s | %4s" "$name" "$xcell" "$gcell" "${xoc:--}" "${GPU_POOL:-4}")
  echo "$row" > "$base.row"
  echo "$row"
done

# collate every <base>.row in this directory into the SUMMARY (sorted) -- fills gaps from prior runs
{ echo "$HDR"; printf '%.0s-' {1..114}; echo; cat "$OUTDIR"/*.row 2>/dev/null | sort; } > "$SUMMARY"
echo; echo "Summary (collated from <base>.row, sorted): $SUMMARY"
echo "Fill failed gaps later:  GPU_POOL=8 ./run_amgx_vs_gpu.sh $OUTDIR/<failed-base> ...   (re-collates automatically)"
