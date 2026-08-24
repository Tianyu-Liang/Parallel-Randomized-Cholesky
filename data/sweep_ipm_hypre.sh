#!/usr/bin/env bash
# Sweep the HARD IPM candidates (uni_chimera, eps0.0001) through HyPre BoomerAMG-PCG.
# Each matrix is run NRUNS times so we can tell a CONSISTENT failure from HyPre's
# 128-thread non-determinism (the same matrix can give 8e-8 one run, 2e-6 the next).
#
# RUN ON A COMPUTE NODE (your salloc) -- NOT on a login node:
#   salloc -N 1 -C cpu -q interactive -t 60:00 -A <acct>   # (or -C gpu, the node has CPUs)
#   ./sweep_ipm_hypre.sh                      # default: all 27 eps0.0001 candidates, 3 runs each
#   NRUNS=5 ./sweep_ipm_hypre.sh              # more repeats
#   ./sweep_ipm_hypre.sh ipmMat/uni_chimera.n100000.*.eps0.001.*.mm   # a different tier
#
# Disk: keeps the IJ conversions in ipm_ij/ (~1 GB for the 27 eps0.0001 matrices).
#       Set DELETE_IJ=1 to delete each matrix's IJ right after its runs (near-zero disk).
#
# Env: NRUNS TOL MAXIT LAUNCHER IJ CPY OUTDIR SEED DELETE_IJ
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA=$SCRIPT_DIR
IJ=${IJ:-ij}
CPY=${CPY:-python3}
OUTDIR=${OUTDIR:-$DATA/ipm_ij}
TOL=${TOL:-1e-8}
MAXIT=${MAXIT:-1000}
NRUNS=${NRUNS:-3}
LAUNCHER=${LAUNCHER:-srun -n 1}          # set LAUNCHER="" to run the ij binary directly
SEED=${SEED:-0}
DELETE_IJ=${DELETE_IJ:-0}
OMP=${OMP:-1}                            # OpenMP threads. 1 = DETERMINISTIC HyPre (reproducible,
                                         #   isolates real failures). >1 is non-deterministic: the
                                         #   parallel reductions give run-to-run residual variation
                                         #   (that's what NRUNS>1 was guarding against). With OMP=1
                                         #   the runs are identical, so NRUNS=1 is enough.
export OMP_NUM_THREADS="$OMP"

cd "$DATA"
mkdir -p "$OUTDIR"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  (1 = deterministic; the HyPre log echoes 'Num OpenMP threads')"

if [ "$#" -gt 0 ]; then
  MATS=("$@")
else
  MATS=( ipmMat/uni_chimera.n100000.*.eps0.0001.*.mm )   # the 27 hard candidates
fi

SUMMARY="$OUTDIR/SWEEP_ipm_hypre_tol${TOL}.txt"
printf "%-38s %7s  %-16s %6s  %s\n" "matrix" "fail/N" "relres_min..max" "it(1)" "verdict" | tee "$SUMMARY"
printf '%.0s-' {1..92} | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"

for MM in "${MATS[@]}"; do
  [ -f "$MM" ] || { echo "  (missing: $MM)"; continue; }
  name=$(basename "$MM" .mm); base="$OUTDIR/$name"

  # convert .mm -> HyPre IJ once (idempotent); RHS = M g / ||M g|| (paper recipe)
  if [ ! -f "$base.00000" ] || [ ! -f "${base}_rhs.00000" ]; then
    "$CPY" convert_ipm_mm_to_ij.py "$MM" "$base" "$SEED" >/dev/null 2>&1 || {
      printf "%-38s %7s  %-16s %6s  %s\n" "$name" "-" "-" "-" "CONVERT_FAIL" | tee -a "$SUMMARY"; continue; }
  fi

  fails=0; rmin=""; rmax=""; it1="-"
  for run in $(seq 1 "$NRUNS"); do
    log="$base.run${run}.log"
    $LAUNCHER "$IJ" -solver 1 -fromfile "$base" -rhsfromfile "${base}_rhs" \
        -tol "$TOL" -max_iter "$MAXIT" -mg_max_iter "$MAXIT" > "$log" 2>&1   # -mg_max_iter is the real AMG-PCG cap
    rr=$(grep -oP 'Final Relative Residual Norm\s*=\s*\K[0-9.eE+-]+' "$log" | tail -1)
    it=$(grep -oP 'Iterations\s*=\s*\K[0-9]+' "$log" | tail -1)
    [ "$run" = 1 ] && it1="${it:--}"
    [ -z "$rr" ] && rr=9.9e99                                    # crash/no-output -> counts as fail
    awk -v r="$rr" -v t="$TOL" 'BEGIN{exit !(r>t)}' && fails=$((fails+1))
    rmin=$(awk -v a="$rmin" -v b="$rr" 'BEGIN{print (a==""||b<a)?b:a}')
    rmax=$(awk -v a="$rmax" -v b="$rr" 'BEGIN{print (a==""||b>a)?b:a}')
  done
  [ "$DELETE_IJ" = 1 ] && rm -f "$base.00000" "${base}_rhs.00000"

  verdict=$(awk -v f="$fails" -v n="$NRUNS" 'BEGIN{
    if(f==0) print "PASS";
    else if(f==n) print "FAIL (consistent)";
    else print "FLAKY ("f"/"n")"}')
  printf "%-38s %7s  %-16s %6s  %s\n" "$name" "$fails/$NRUNS" \
    "$(awk -v a="$rmin" -v b="$rmax" 'BEGIN{printf "%.1e..%.1e", a, b}')" "$it1" "$verdict" | tee -a "$SUMMARY"
done
echo; echo "Summary: $SUMMARY   (per-run logs: ipm_ij/<name>.run<k>.log)"
