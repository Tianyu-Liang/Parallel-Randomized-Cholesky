#!/usr/bin/env bash
# AMGX (classical AMG-preconditioned PCG) on the SAME chimera matrices as HyPre/AC.
# AMGX, like HyPre, solves the RAW matrix directly (no augmentation) with the SAME b=M*g.
#
# Per matrix it needs the AMGX-format system <base>.amgx, produced by gen_amgx_from_ij.py
# from the HyPre IJ files (<base>.00000 + <base>_rhs.00000) -- so it's the identical
# (matrix, RHS) the other solvers use. The script auto-converts any missing .amgx.
#
# RUN inside a GPU allocation (AMGX is GPU-only):
#   salloc -N 1 -C gpu -q interactive -t 60:00 -A <acct>
#   ./run_amgx_chimera.sh chimera_amd                 # a directory: all *.00000 systems
#   ./run_amgx_chimera.sh chimera_amd/uni_chimera.n100000.s12   # one base (no extension)
#
# Beyond it/relres, it records Operator Complexity and peak memory -- the AMG fill/memory
# blowup that is the point of the bounded-memory comparison.
#
# Env: TOL AMGX CFG MODE AMGX_LIB
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
AMGX=${AMGX:-amgx_capi}
CFG=${CFG:-$ROOT/data/amgx_pcg_amg_t1e8.json}
MODE=${MODE:-dDDI}                  # device, double matrix, double vector, int index
AMGX_LIB=${AMGX_LIB:-}
TOL=${TOL:-1e-8}
CONV=$ROOT/data/gen_amgx_from_ij.py
[ -z "$AMGX_LIB" ] || export LD_LIBRARY_PATH="$AMGX_LIB:${LD_LIBRARY_PATH:-}"

# verdict: PASS if the (relative) residual reached ~tol, WEAK if it at least made progress,
# FAIL if it crashed or stalled far from tol. 10x allowance for monitor-vs-true wiggle.
classify() { awk -v r="$1" -v t="$TOL" 'BEGIN{
  if(r=="-"){print "FAIL(crash)";exit}
  f=r/t; print (f<=1e1)?"PASS":(r<=1e-2)?"WEAK":"FAIL"}'; }

# collect bases (strip .00000 / .amgx; a dir -> every *.00000 system)
declare -a BASE=()
for arg in "$@"; do
  if [ -d "$arg" ]; then
    while IFS= read -r f; do BASE+=("${f%.00000}"); done \
      < <(find "$arg" -name '*.00000' ! -name '*_rhs.00000' | sort)
  else
    BASE+=("${arg%.00000}"); BASE[-1]="${BASE[-1]%.amgx}"
  fi
done
[ "${#BASE[@]}" -eq 0 ] && { echo "no systems found"; exit 1; }

OUTDIR=$(dirname "${BASE[0]}")
SUMMARY="$OUTDIR/SUMMARY_amgx_tol${TOL}.txt"
printf "%-42s | %-26s | %8s | %9s\n" "matrix" "AMGX(it/relres/verdict)" "op_cmplx" "hierMem" | tee "$SUMMARY"
printf '%.0s-' {1..96} | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"

for base in "${BASE[@]}"; do
  name=$(basename "$base")
  [ -f "$base.amgx" ] || python3 "$CONV" "$base" >/dev/null 2>&1
  log="$base.amgx.log"
  $AMGX -mode "$MODE" -m "$base.amgx" -c "$CFG" > "$log" 2>&1
  it=$(grep -oP 'Total Iterations:\s*\K[0-9]+' "$log" | tail -1)
  rr=$(grep -oP 'Final Residual:\s*\K[0-9.eE+-]+' "$log" | tail -1)
  oc=$(grep -oP 'Operator Complexity:\s*\K[0-9.eE+-]+' "$log" | tail -1)
  # hierarchy footprint (the AMG fill memory) -- NOT "Maximum Memory Usage", which is AMGX's
  # pre-allocated device pool (~GPU size) and is the same for every matrix.
  mem=$(grep -oP 'Total Memory Usage:\s*\K[0-9.]+' "$log" | tail -1)
  cell=$(printf "%s/%s/%s" "${it:--}" "${rr:--}" "$(classify "${rr:--}")")
  printf "%-42s | %-26s | %8s | %8sG\n" "$name" "$cell" "${oc:--}" "${mem:--}" | tee -a "$SUMMARY"
done
echo; echo "Summary: $SUMMARY   (per-matrix logs: <base>.amgx.log)"
