
export OMP_PROC_BIND="close"
export MKL_NUM_THREADS="4"
export KMP_AFFINITY="norespect"

list1=("../physics/parabolic_fem/parabolic_fem-nnz-sorted.mtx"  "../physics/ecology2/ecology2-nnz-sorted.mtx" "../physics/apache2/apache2-nnz-sorted.mtx" "../physics/G3_circuit/G3_circuit-nnz-sorted.mtx" "../physics/uniform_3D/uniform_3D-nnz-sorted.mtx" "../physics/aniso_contrast_3D/aniso_contrast_3D-nnz-sorted.mtx")
list2=("../physics/parabolic_fem/"  "../physics/ecology2/" "../physics/apache2/" "../physics/G3_circuit/" "../physics/uniform_3D/" "../physics/aniso_contrast_3D/")
# list1=("../physics/poisson_contrast_3D/poisson_contrast_3D-nnz-sorted.mtx")
# list2=("../physics/poisson_contrast_3D/")

for (( i=0; i<${#list1[@]}; i++ )); do
  number=1
  # Loop until the number exceeds a chosen limit (e.g., 16)
  while [ $number -le 32 ]; do
    ../cpu_implementation/driver ${list1[i]} $number "" 1
    number=$(( number * 2 ))
  done
  ./driver ${list1[i]} 32 "" 1
done




