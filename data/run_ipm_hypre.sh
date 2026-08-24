#!/usr/bin/env bash
# Reproduce HyPre BoomerAMG-PCG failure on the ipmMat (Flow IPM) matrices.
#
# Per arXiv:2303.00709: target the uni_chimera.* matrices (NOT spielman, which
# HyPre solves), tolerance = 1e-8 relative residual, RHS = M g/||M g||.
# A "failure" = final relative residual stays above 1e-8.
#
# MUST be run inside a SLURM allocation (login nodes hang / are off-limits):
#   salloc -N 1 -C gpu -t 30:00 -q interactive -A <account>
#   ./run_ipm_hypre.sh                       # default curated sweep over eps
#   ./run_ipm_hypre.sh ipmMat/uni_chimera.n100000.i1.eps0.0001.2.mm ...   # explicit list
#
# Env overrides: TOL, MAXIT, LAUNCHER, IJ, OUTDIR, CPY, SEED
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA=$SCRIPT_DIR
IJ=${IJ:-ij}
CPY=${CPY:-python3}
OUTDIR=${OUTDIR:-$DATA/ipm_ij}
TOL=${TOL:-1e-8}
MAXIT=${MAXIT:-1000}
LAUNCHER=${LAUNCHER:-srun -n 1}
SEED=${SEED:-0}

cd "$DATA"
mkdir -p "$OUTDIR"

# Default curated sweep: one instance per eps, spanning easy->hard (small eps = hard).
if [ "$#" -gt 0 ]; then
  MATS=("$@")
else
  MATS=(
    ipmMat/uni_chimera.n100000.i1.eps0.1.2.mm
    ipmMat/uni_chimera.n100000.i1.eps0.01.2.mm
    ipmMat/uni_chimera.n100000.i1.eps0.001.2.mm
    ipmMat/uni_chimera.n100000.i1.eps0.0001.2.mm
    ipmMat/uni_chimera.n100000.i3.eps0.0001.2.mm
    ipmMat/uni_chimera.n100000.i5.eps0.0001.2.mm
  )
fi

SUMMARY="$OUTDIR/SUMMARY_tol${TOL}.txt"
printf "%-52s %8s %8s %14s   %s\n" "matrix" "iters" "setup_s" "final_relres" "verdict" | tee "$SUMMARY"
printf '%.0s-' {1..100} | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"

for MM in "${MATS[@]}"; do
  name=$(basename "$MM" .mm)
  base="$OUTDIR/$name"

  # convert if not already present (idempotent)
  if [ ! -f "$base.00000" ] || [ ! -f "${base}_rhs.00000" ]; then
    "$CPY" convert_ipm_mm_to_ij.py "$MM" "$base" "$SEED" >/dev/null || {
      printf "%-52s %8s %8s %14s   %s\n" "$name" "-" "-" "-" "CONVERT_FAIL" | tee -a "$SUMMARY"; continue; }
  fi

  log="$OUTDIR/$name.log"
  $LAUNCHER "$IJ" -solver 1 -fromfile "$base" -rhsfromfile "${base}_rhs" \
      -tol "$TOL" -max_iter "$MAXIT" > "$log" 2>&1

  iters=$(grep -oP 'Iterations\s*=\s*\K[0-9]+' "$log" | tail -1)
  relres=$(grep -oP 'Final Relative Residual Norm\s*=\s*\K[0-9.eE+-]+' "$log" | tail -1)
  setup=$(grep -A4 'PCG Setup' "$log" | grep -oP 'wall clock time\s*=\s*\K[0-9.]+' | tail -1)
  [ -z "$iters" ] && iters="-"
  [ -z "$setup" ] && setup="-"

  if [ -z "$relres" ]; then
    verdict="INF(crash/no-output)"; relres="-"
  else
    # factor = relres / 1e-8 ; classify like the paper
    verdict=$(awk -v r="$relres" 'BEGIN{
      t=1e-8; f=r/t;
      if (f<=1)        print "PASS";
      else if (f<=1e4) print "FAIL*  (>1e-8)";
      else if (f<=1e8) print "FAIL** (>1e-4)";
      else             print "INF    (>1)";
    }')
  fi
  printf "%-52s %8s %8s %14s   %s\n" "$name" "$iters" "$setup" "$relres" "$verdict" | tee -a "$SUMMARY"
done

echo
echo "Summary written to: $SUMMARY"
echo "Per-matrix HyPre logs in: $OUTDIR/<name>.log"
