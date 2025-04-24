
export OMP_PROC_BIND=close
export MKL_NUM_THREADS="4"
export KMP_AFFINITY=norespect

list1=("../physics/spe0.5m/spe0.5m-nnz-sorted.mtx" "../physics/spe2m/spe2m-nnz-sorted.mtx" "../physics/spe4m/spe4m-nnz-sorted.mtx" "../physics/spe8m/spe8m-nnz-sorted.mtx" "../physics/spe16m/spe16m-nnz-sorted.mtx")
list2=("../physics/spe0.5m/" "../physics/spe2m/" "../physics/spe4m/" "../physics/spe8m/" "../physics/spe16m/")

for (( i=0; i<${#list1[@]}; i++ )); do
  number=1
  # Loop until the number exceeds a chosen limit (e.g., 16)
  while [ $number -le 32 ]; do
    ../cpu_implementation/driver ${list1[i]} $number "" 1
    number=$(( number * 2 ))
  done
  ./driver ${list1[i]} 32 "" 1
done





