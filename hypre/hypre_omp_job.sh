
cd data_dir/mats


# Thread counts to test
THREADS=(1 2 4 8 16 32)

# Matrix files and their corresponding RHS files (filenames only)
MATRICES=(
  "aniso-contrast_1" "aniso-contrast_rhs.txt"
  "apache2_1" "apache2_rhs.txt"
  "belgium_osm_lap_1" "belgium_osm_lap_rhs.txt"
  "com-LiveJournal_lap_1" "com-LiveJournal_lap_rhs.txt"
  "delaunay_n24_lap_1" "delaunay_n24_lap_rhs.txt"
  "ecology1_1" "ecology1_rhs.txt"
  "ecology2_1" "ecology2_rhs.txt"
  "europe_osm_lap_1" "europe_osm_lap_rhs.txt"
  "G3_circuit_1" "G3_circuit_rhs.txt"
  "GAP-road_lap_1" "GAP-road_lap_rhs.txt"
  "parabolic_fem_1" "parabolic_fem_rhs.txt"
  "poisson-contrast_1" "poisson-contrast_rhs.txt"
  "spe16_1" "spe16_rhs.txt"
  "uniform3D_1" "uniform3D_rhs.txt"
  "venturiLevel3_lap_1" "venturiLevel3_lap_rhs.txt"
)


# Loop through thread counts
for nt in "${THREADS[@]}"; do
  export OMP_NUM_THREADS=$nt
  echo "Running with $nt threads:"

  # Loop through matrices
  for i in $(seq 0 $((${#MATRICES[@]} / 2 - 1))); do
    matrix="${MATRICES[i*2]}"
    rhs="${MATRICES[i*2+1]}"

    # Construct the output log filename
    log_file="omplogs/${matrix}_${nt}threads.log"

    # Run the Hypre solver with timeout.  Filenames are relative to MAT_DIR.
    timeout 3000 srun -n 1 -c $nt ../hypre/src/test/ij -solver 1 -fromfile "$matrix" -rhsfromonefile "$rhs" -tol 1.0e-6 > "$log_file" 2>&1

    echo "  Finished $matrix with $nt threads. Log: $log_file"
  done
done

echo "All runs completed."
