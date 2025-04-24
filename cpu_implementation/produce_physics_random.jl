include("write_graph.jl")
include("lap_grid.jl")

physics_produce("../physics/parabolic_fem/parabolic_fem", "random")
physics_produce("../physics/parabolic_fem/parabolic_fem", "random")
physics_produce("../physics/ecology1/ecology1", "random")
physics_produce("../physics/ecology2/ecology2", "random")
physics_produce("../physics/apache2/apache2", "random")
physics_produce("../physics/G3_circuit/G3_circuit", "random")
uniform_grid_sddm(10000) # warm up
uniform_produce("../physics/uniform_3D/uniform_3D", 100000000, "random")
wgrid_sddm(10000, 10) #warm up
aniso_contrast_produce("../physics/aniso_contrast_3D/aniso_contrast_3D", 100000000, 10000, "random")

checkered_grid_sddm(1000, 2, 2, 2, 1);
poisson_contrast_produce("../physics/poisson_contrast_3D/poisson_contrast_3D", 100000000, 64, 1e7, "random");