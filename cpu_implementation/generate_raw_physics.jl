include("write_graph.jl")
include("lap_grid.jl")

# Generate raw matrices without any ordering, just save the base .mtx
# These are used by generate_solver_inputs.py to create hypre/amgx inputs

# uniform_3D: 100M nnz target
println("Generating uniform_3D...")
G = uniform_grid_sddm(100000000)
println("  size: ", size(G, 1), " x ", size(G, 2), " nnz: ", nnz(G))
MatrixMarket.mmwrite("../physics/uniform_3D/uniform_3D.mtx", G)
println("  written to ../physics/uniform_3D/uniform_3D.mtx")
GC.gc()

# aniso_contrast_3D: 100M nnz target, contrast factor 10000
println("Generating aniso_contrast_3D...")
wgrid_sddm(10000, 10) # warm up
G = wgrid_sddm(100000000, 10000)
println("  size: ", size(G, 1), " x ", size(G, 2), " nnz: ", nnz(G))
MatrixMarket.mmwrite("../physics/aniso_contrast_3D/aniso_contrast_3D.mtx", G)
println("  written to ../physics/aniso_contrast_3D/aniso_contrast_3D.mtx")
GC.gc()

# poisson_contrast_3D: 100M nnz target, grid_len=64, contrast=1e7
println("Generating poisson_contrast_3D...")
checkered_grid_sddm(1000, 2, 2, 2, 1) # warm up
G = checkered_grid_sddm(100000000, 64, 64, 64, 1e7)
println("  size: ", size(G, 1), " x ", size(G, 2), " nnz: ", nnz(G))
MatrixMarket.mmwrite("../physics/poisson_contrast_3D/poisson_contrast_3D.mtx", G)
println("  written to ../physics/poisson_contrast_3D/poisson_contrast_3D.mtx")
GC.gc()

println("All done!")
