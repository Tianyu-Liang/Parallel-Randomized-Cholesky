
export OMP_PROC_BIND="close"
export MKL_NUM_THREADS="4"
export KMP_AFFINITY="norespect"


list1=("../physics/ecology1/ecology1-rand.mtx" "../data/GAP-road/GAP-road-rand.mtx" "../data/europe_osm/europe_osm-rand.mtx" "../data/com-LiveJournal/com-LiveJournal-rand.mtx" "../data/delaunay_n24/delaunay_n24-rand.mtx" "../data/venturiLevel3/venturiLevel3-rand.mtx" "../data/belgium_osm/belgium_osm-rand.mtx")
list2=("../physics/ecology1/" "../data/GAP-road/" "../data/europe_osm/" "../data/com-LiveJournal/" "../data/delaunay_n24/" "../data/venturiLevel3/" "../data/belgium_osm/")
# "../data/europe_osm/europe_osm-rand.mtx" "../data/europe_osm/" 
# list1=("../data/europe_osm/europe_osm-rand.mtx" "../data/delaunay_n24/delaunay_n24-rand.mtx")
# list2=("../data/europe_osm/" "../data/delaunay_n24/")
# list1=("../data/venturiLevel3/venturiLevel3-rand.mtx" "../data/roadNet-TX/roadNet-TX-rand.mtx" "../data/belgium_osm/belgium_osm-rand.mtx")
# list2=("../data/venturiLevel3/"  "../data/roadNet-TX/" "../data/belgium_osm/")

for (( i=0; i<${#list1[@]}; i++ )); do
  number=1
  # Loop until the number exceeds a chosen limit (e.g., 16)
  while [ $number -le 32 ]; do
    ../cpu_implementation/driver ${list1[i]} $number ""
    number=$(( number * 2 ))
  done
  ./driver ${list1[i]} 32 ""
done




