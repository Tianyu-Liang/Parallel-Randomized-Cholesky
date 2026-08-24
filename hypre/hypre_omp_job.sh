#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DATA=${DATA:-$ROOT/data/hypre}
IJ=${IJ:-ij}
LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/omplogs}
mkdir -p "$LOG_DIR"

# Thread counts to test
THREADS=(1 2 4 8 16 32)

# Matrix names
MATRICES=(
  aniso_contrast_3D
  apache2
  belgium_osm
  com-LiveJournal
  delaunay_n24
  ecology1
  ecology2
  europe_osm
  G3_circuit
  GAP-road
  parabolic_fem
  poisson_contrast_3D
  spe16m
  uniform_3D
  venturiLevel3
)

# Loop through thread counts
for nt in "${THREADS[@]}"; do
  export OMP_NUM_THREADS=$nt
  echo "Running with $nt threads:"

  for matrix in "${MATRICES[@]}"; do
    log_file="$LOG_DIR/${matrix}_${nt}threads.log"

    timeout 3000 srun -n 1 -c $nt $IJ -solver 1 -fromfile $DATA/$matrix -rhsfromfile $DATA/${matrix}_rhs -tol 1.0e-6 > "$log_file" 2>&1

    echo "  Finished $matrix with $nt threads. Log: $log_file"
  done
done

echo "All runs completed."
