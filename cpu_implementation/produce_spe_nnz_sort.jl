include("write_graph.jl")
include("lap_grid.jl")

# first run warm up
physics_produce("../physics/spe0.5m/spe0.5m", "nnz-sort", negate=true)
physics_produce("../physics/spe0.5m/spe0.5m", "nnz-sort", negate=true)
physics_produce("../physics/spe2m/spe2m", "nnz-sort", negate=true)
physics_produce("../physics/spe4m/spe4m", "nnz-sort", negate=true)
physics_produce("../physics/spe8m/spe8m", "nnz-sort", negate=true)
physics_produce("../physics/spe16m/spe16m", "nnz-sort", negate=true)
