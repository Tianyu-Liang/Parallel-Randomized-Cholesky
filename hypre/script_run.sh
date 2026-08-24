#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DATA=${DATA:-$ROOT/data/hypre}
IJ=${IJ:-ij}

# Graph Laplacians
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/com-LiveJournal -rhsfromfile $DATA/com-LiveJournal_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/GAP-road -rhsfromfile $DATA/GAP-road_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/europe_osm -rhsfromfile $DATA/europe_osm_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/delaunay_n24 -rhsfromfile $DATA/delaunay_n24_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/venturiLevel3 -rhsfromfile $DATA/venturiLevel3_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/belgium_osm -rhsfromfile $DATA/belgium_osm_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}

# Physics matrices
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/parabolic_fem -rhsfromfile $DATA/parabolic_fem_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/ecology1 -rhsfromfile $DATA/ecology1_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/ecology2 -rhsfromfile $DATA/ecology2_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/apache2 -rhsfromfile $DATA/apache2_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/G3_circuit -rhsfromfile $DATA/G3_circuit_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/spe16m -rhsfromfile $DATA/spe16m_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/uniform_3D -rhsfromfile $DATA/uniform_3D_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/aniso_contrast_3D -rhsfromfile $DATA/aniso_contrast_3D_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
srun -n 1 -c $OMP_NUM_THREADS $IJ -fromfile $DATA/poisson_contrast_3D -rhsfromfile $DATA/poisson_contrast_3D_rhs -solver 1 -tol 1e-6
printf '\n%.0s' {1..30}
