# Instructions
"cpu_implementation" contains the cpu version of our code, and it contains only the factorization part. "experiment" folder contains the complete pipeline (factorization and solve on cpu). "gpu_implementation" contains the gpu code.

The user needs to first download fast matrix market from "https://github.com/alugowski/fast_matrix_market"
To run the code, we would first need to download the relevant matrices from suitesparse and put them into a folder (e.g. data, physics). For instance, the location of "parabolic_fem" should be "data/parabolic_fem/parabolic_fem.mtx".
Then one should use write_graph.jl from "cpu_implementation" to write down the reordered matrices. See the jl files starting with the prefix "produce" for examples on how to create those matrices.
**Make sure to create a folder for each matrix (i.e. parabolic_fem folder would contain all variations of the parabolic_fem matrices). Folders must be manually created for matrices that are not from SuiteSparse (i.e. 3D uniform poisson, etc.)**

Lastly, fill out the correct paths in the makefile and compile the code. 

**CPU**:
For matrices that are not originally Laplacian (i.e. sddm converted to a laplacian by appending a row and a column to the end), run with 
```console
$ ./driver path/to/parabolic_fem-nnz-sorted.mtx 32 "" 1
```
The second input is the matrix path. The third input indicates the number of threads to use. The 4th input indicates the location to write the computed factorization to. An empty string for the 4th input tells the program to not write anything. The last flag "1" simply indicates that this matrix is not originally a Laplacian, and the true solution will need to be trimmed and converted. 
If the matrix is already a graph Laplacian, then simply remove the last flag, such as: 
```console
$ ./driver path/to/parabolic_fem-nnz-sorted.mtx 32 ""
```
Also see physics_test_nnz_sort.sh and similar files for examples.

**The code in experiment folder requires MKL. On Perlmutter, the code can be compiled by using "module load intel" to load the intel paths.**

**GPU**
The there are two relevant files in gpu_implementation that handles both the natural Laplacian (driver.cu) and converted SDDM to Laplacians (driver_physics.cu) respectively. One can change which file to compile in the makefile. An example run is the follow:
```console
$ ./driver path/to/parabolic_fem-nnz-sorted.mtx 512 1 7e-7
```
The 3rd input is the number of thread blocks to use. The fourth input indicates whether the solver part should be ran (0 indicates skip, 1 indicates run). The last input is a number in scientific notation that indicates the desired accuracy of the solution.
Also see test_script_physics_nnz_sort and similar files for examples.

