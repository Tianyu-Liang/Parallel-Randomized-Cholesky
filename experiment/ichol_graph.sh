
export OMP_PROC_BIND="close"
export MKL_NUM_THREADS="4"
export KMP_AFFINITY="norespect"

./independent_cg "../physics/ecology1/ecology1-amd.mtx" "../physics/ecology1/_ic_amd.mtx" 1
./independent_cg "../data/GAP-road/GAP-road-amd.mtx" "../data/GAP-road/_ic_amd.mtx" 1
./independent_cg "../data/com-LiveJournal/com-LiveJournal-amd.mtx" "../data/com-LiveJournal/_ic_amd.mtx" 1
./independent_cg "../data/europe_osm/europe_osm-amd.mtx" "../data/europe_osm/_ic_amd.mtx" 1
./independent_cg "../data/delaunay_n24/delaunay_n24-amd.mtx" "../data/delaunay_n24/_ic_amd.mtx" 1
./independent_cg "../data/venturiLevel3/venturiLevel3-amd.mtx" "../data/venturiLevel3/_ic_amd.mtx" 1
./independent_cg "../data/belgium_osm/belgium_osm-amd.mtx" "../data/belgium_osm/_ic_amd.mtx" 1