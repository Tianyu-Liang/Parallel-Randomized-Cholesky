#!/usr/bin/env bash
# Generic HYPRE AMG-PCG runner over pre-generated IJ files (<base>.00000 + <base>_rhs.00000).
# Runs -solver 1 at tol 1e-8, parses the final residual, flags CG breakdown (negative
# alpha), and classifies pass/fail with the paper's thresholds (arXiv:2303.00709).
#
# Must run inside a SLURM allocation (solves don't belong on login nodes).
#   ./run_hypre_ij.sh chimera_ij                          # whole dir (only chimera files there)
#   ./run_hypre_ij.sh chimera_ij/uni_chimera.*.00000      # only the uni_chimera matrices
#   ./run_hypre_ij.sh ipm_ij/uni_chimera.*eps0.0001*.00000  # any other IJ set works too
#
# Env: TOL MAXIT LAUNCHER IJ
set -u
IJ=${IJ:-ij}
TOL=${TOL:-1e-8}
MAXIT=${MAXIT:-1000}
LAUNCHER=${LAUNCHER:-srun -n 1}

# collect base paths (strip .00000), excluding *_rhs.00000
declare -a BASES=()
for arg in "$@"; do
  if [ -d "$arg" ]; then
    while IFS= read -r f; do BASES+=("${f%.00000}"); done \
      < <(find "$arg" -name '*.00000' ! -name '*_rhs.00000' | sort)
  else
    case "$arg" in *_rhs.00000) continue;; esac   # skip rhs files if a glob picked them up
    BASES+=("${arg%.00000}")
  fi
done
[ "${#BASES[@]}" -eq 0 ] && { echo "no IJ files found"; exit 1; }

OUTDIR=$(dirname "${BASES[0]}")
SUMMARY="$OUTDIR/SUMMARY_hypre_tol${TOL}.txt"
printf "%-46s %6s %6s %14s  %-18s %s\n" "instance" "iters" "setup" "final_relres" "verdict" "breakdown" | tee "$SUMMARY"
printf '%.0s-' {1..100} | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"

for base in "${BASES[@]}"; do
  name=$(basename "$base")
  [ -f "$base.00000" ] && [ -f "${base}_rhs.00000" ] || { printf "%-46s  MISSING files\n" "$name" | tee -a "$SUMMARY"; continue; }
  log="$base.hypre.log"
  $LAUNCHER "$IJ" -solver 1 -fromfile "$base" -rhsfromfile "${base}_rhs" \
      -tol "$TOL" -max_iter "$MAXIT" > "$log" 2>&1

  iters=$(grep -oP 'Iterations\s*=\s*\K[0-9]+' "$log" | tail -1)
  relres=$(grep -oP 'Final Relative Residual Norm\s*=\s*\K[0-9.eE+-]+' "$log" | tail -1)
  setup=$(grep -A4 'PCG Setup' "$log" | grep -oP 'wall clock time\s*=\s*\K[0-9.]+' | tail -1)
  brk=$(grep -qE 'alpha +-' "$log" && echo "NEG-ALPHA" || echo "-")
  [ -z "$iters" ] && iters="-"; [ -z "$setup" ] && setup="-"

  if [ -z "$relres" ]; then verdict="INF(crash)"; relres="-"
  else verdict=$(awk -v r="$relres" 'BEGIN{t=1e-8;f=r/t;
        print (f<=1)?"PASS":(f<=1e4)?"FAIL*(>1e-8)":(f<=1e8)?"FAIL**(>1e-4)":"INF(>1)"}'); fi
  printf "%-46s %6s %6s %14s  %-18s %s\n" "$name" "$iters" "$setup" "$relres" "$verdict" "$brk" | tee -a "$SUMMARY"
done
echo; echo "Summary: $SUMMARY   (per-run logs: <base>.hypre.log)"
