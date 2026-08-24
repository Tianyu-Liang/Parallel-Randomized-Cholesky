#!/usr/bin/env bash
# Head-to-head: ParAC (CPU driver) vs HyPre BoomerAMG-PCG on the SAME matrices.
# Goal: show AC converges where HyPre diverges/stalls (the robustness advantage).
#
# Uses the CPU AC driver (experiment/driver, built from driver_local.cpp) so there's NO fixed
# GPU pool to overflow on dense matrices (it grows storage dynamically).
#
# Per matrix it expects:
#   <base>.mtx                       -> AC driver  (MatrixMarket; gen_chimera_sweep.jl writes it;
#                                       the ipmMat/*.mm files are already this format)
#   <base>.00000 + <base>_rhs.00000  -> HyPre ij   (optional; skipped if absent)
#
# RUN inside a CPU allocation, AND `module load intel` first (the CPU AC driver links MKL):
#   salloc -N 1 -C cpu -q interactive -t 60:00 -A <acct>
#   module load intel
#   ./run_ac_vs_hypre.sh chimera_ij/uni_chimera.n100000.s32.mtx chimera_ij/uni_chimera.n100000.s44.mtx
#   ./run_ac_vs_hypre.sh chimera_ij              # a directory: globs *.mtx
#
# CAVEATS:
#  - AC now reads the same <base>_rhs.00000 (b = M*g) that HyPre uses, in graph mode (full matrix,
#    no node removal), so the two solve the IDENTICAL (matrix, RHS). [Was previously a zero-sum
#    random RHS on a node-removed matrix -- not comparable.]
#  - "relative residual" (AC) and HyPre's reported residual differ from the recursive stopping
#    residual by the CG residual gap, so PASS allows up to 1000x tol. A value just above tol is
#    likely the gap, not a real failure; only WEAK/FAIL (>>tol, up to >1e-2) is a genuine miss.
#  - AC's CPU solve only runs when num_threads == 32 (a hardcoded gate in driver_local.cpp), so
#    NTHREADS must stay 32.
#
# Env: TOL NTHREADS MAXIT LAUNCHER AC_LAUNCHER AC IJ
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
AC=${AC:-$ROOT/experiment/driver}
IJ=${IJ:-ij}
TOL=${TOL:-1e-8}                      # comparison target (HyPre uses this; verdicts are vs this)
# AC stops on MKL's RECURSIVE residual, which sits ~2-10x above the TRUE ||b-Ax|| (CG residual gap).
# So solve AC to a slightly tighter recursive tol than TOL, so its TRUE residual lands at ~TOL.
# (Logic in custom_cg.hpp is unchanged; this only passes a tighter target. Dial AC_TOL if needed.)
AC_TOL=${AC_TOL:-1e-9}
NTHREADS=${NTHREADS:-32}              # must be 32 (driver_local.cpp gates the solve on this)
HYPRE_OMP=${HYPRE_OMP:-1}             # OpenMP threads for HyPre ONLY. 1 = DETERMINISTIC (reproducible
                                     #   failures). Unset/>1 uses all cores (256) -> non-deterministic
                                     #   residuals. AC is unaffected -- it keeps NTHREADS (32).
MAXIT=${MAXIT:-1000}                  # HyPre iteration cap. NOTE: for -solver 1 (AMG-PCG) the Krylov
                                     #   loop is capped by -mg_max_iter (ij.c default 100!), NOT
                                     #   -max_iter -- so we pass BOTH below to actually get 1000.
LAUNCHER=${LAUNCHER:-srun -n 1}       # HyPre
AC_LAUNCHER=${AC_LAUNCHER:-}          # AC: empty = run directly (inherits the module-loaded MKL env)

# verdict relative to tol, with a 1000x gap allowance for the recursive-vs-true residual gap
classify() { awk -v r="$1" -v t="$TOL" 'BEGIN{
  if(r=="-"){print "FAIL(crash)";exit}
  f=r/t;
  print (f<=1e3)?"PASS":(r<=1e-2)?"WEAK":"FAIL"}'; }

# collect absolute .mtx paths
declare -a MTX=()
for arg in "$@"; do
  if [ -d "$arg" ]; then
    while IFS= read -r f; do MTX+=("$(readlink -f "$f")"); done < <(find "$arg" -name '*.mtx' | sort)
  else
    MTX+=("$(readlink -f "$arg")")
  fi
done
[ "${#MTX[@]}" -eq 0 ] && { echo "no .mtx files found"; exit 1; }

OUTDIR=$(dirname "${MTX[0]}")
SUMMARY="$OUTDIR/SUMMARY_ac_vs_hypre_tol${TOL}.txt"
printf "%-42s | %-24s | %-22s\n" "matrix" "HyPre(it/relres/verdict)" "AC(it/relres/verdict)" | tee "$SUMMARY"
printf '%.0s-' {1..96} | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"

for mtx in "${MTX[@]}"; do
  base="${mtx%.mtx}"; name=$(basename "$base")

  # ---- HyPre (if IJ files present) ----
  if [ -f "$base.00000" ] && [ -f "${base}_rhs.00000" ]; then
    hlog="$base.hypre.log"
    OMP_NUM_THREADS=$HYPRE_OMP $LAUNCHER "$IJ" -solver 1 -fromfile "$base" -rhsfromfile "${base}_rhs" -tol "$TOL" -max_iter "$MAXIT" -mg_max_iter "$MAXIT" > "$hlog" 2>&1
    hit=$(grep -oP 'Iterations\s*=\s*\K[0-9]+' "$hlog" | tail -1)
    hr=$(grep -oP 'Final Relative Residual Norm\s*=\s*\K[0-9.eE+-]+' "$hlog" | tail -1)
    grep -qE 'alpha +-' "$hlog" && hbrk="!brk" || hbrk=""
    hcell=$(printf "%s/%s/%s%s" "${hit:--}" "${hr:--}" "$(classify "${hr:--}")" "$hbrk")
  else
    hcell="(no IJ files)"
  fi

  # ---- AC (CPU driver: <matrix> <num_threads> <path> <is_graph> <tol> [rhsfile]) ----
  # is_graph picks the mode by family (gen_chimera_sweep.jl writes the .mtx to match):
  #   *_bndry_chimera = SDDM -> .mtx is the AUGMENTED Laplacian (size k+1) -> PHYSICS mode (0),
  #                     which trims the appended ground node so the factor G satisfies G^T G ~= M.
  #   *_chimera       = true Laplacian -> .mtx is raw (size k) -> GRAPH mode (1).
  # Feeding a raw SDDM in graph mode silently degrades the preconditioner (see README), so this
  # split is REQUIRED for a fair comparison. The _rhs.00000 (size k) makes AC solve the SAME b=M*g
  # that HyPre does, on the same SOLVED system M.
  case "$name" in *bndry*) is_graph=0;; *) is_graph=1;; esac
  alog="$base.ac.log"
  $AC_LAUNCHER "$AC" "$mtx" "$NTHREADS" "" "$is_graph" "$AC_TOL" "${base}_rhs.00000" > "$alog" 2>&1
  ait=$(grep -oP 'Iterations:\s*\K[0-9]+' "$alog" | tail -1)
  ar=$(grep -oP 'relative residual:\s*\K[0-9.eE+-]+' "$alog" | tail -1)
  acell=$(printf "%s/%s/%s" "${ait:--}" "${ar:--}" "$(classify "${ar:--}")")

  printf "%-42s | %-24s | %-22s\n" "$name" "$hcell" "$acell" | tee -a "$SUMMARY"
done
echo; echo "Summary: $SUMMARY   (logs: <base>.hypre.log / <base>.ac.log)"
