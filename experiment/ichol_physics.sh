
export OMP_PROC_BIND="close"
export MKL_NUM_THREADS="4"
export KMP_AFFINITY="norespect"

./independent_cg "../physics/parabolic_fem/parabolic_fem-amd.mtx" "../physics/parabolic_fem/_ic_amd.mtx" 0
./independent_cg "../physics/ecology2/ecology2-amd.mtx" "../physics/ecology2/_ic_amd.mtx" 0
./independent_cg "../physics/apache2/apache2-amd.mtx" "../physics/apache2/_ic_amd.mtx" 0
./independent_cg "../physics/G3_circuit/G3_circuit-amd.mtx" "../physics/G3_circuit/_ic_amd.mtx" 0
./independent_cg "../physics/uniform_3D/uniform_3D-amd.mtx" "../physics/uniform_3D/_ic_amd.mtx" 0
./independent_cg "../physics/aniso_contrast_3D/aniso_contrast_3D-amd.mtx" "../physics/aniso_contrast_3D/_ic_amd.mtx" 0
./independent_cg "../physics/poisson_contrast_3D/poisson_contrast_3D-amd.mtx" "../physics/poisson_contrast_3D/_ic_amd.mtx" 0
./independent_cg "../physics/spe16m/spe16m-amd.mtx" "../physics/spe16m/_ic_amd.mtx" 0