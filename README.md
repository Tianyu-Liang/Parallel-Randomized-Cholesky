# Software Needed
1. Linux OS (maybe MacOS, not tested yet).
2. Julia programming language is required to run some matrix preprocessing step. The instructions for download can be found on: [Julia Programming Language](https://github.com/JuliaLang/julia).
3. [Fast matrix market](https://github.com/alugowski/fast_matrix_market) is required for reading in matrix market files. This package can be used as a template based library, which means no compilation is needed. The user can simply clone the repo. Later on, user will have to specify the path to the include folder of fast matrix market in makefile.
4. Matlab (optional, used for comparison benchmarks)
5. g++ (version 12 or 13)
6. Intel c++ compiler (icpx or similar, 2024 version) for compiling intel MKL code, which is used by the solver. This is optional and only required if you want to run the complete factorize + solve pipeline in the experiment folder (CPU).
7. CUDA (12.4) for running GPU experiments. An example download instruction for version 12.4 can be found here: [cuda toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive). Other versions starting with 12 may also work.




# Instructions
We provide a series of instructions for running our implementation for a subset of the experiments found in [the paper](https://arxiv.org/abs/2505.02977). These instructions will show you how to download some of the datasets, preprocess them, and run example versions of our experiments.

## Downloading datasets
TODO

"cpu_implementation" contains the cpu version of our code, and it contains only the factorization part. "experiment" folder contains the complete pipeline (factorization and solve on cpu). "gpu_implementation" contains the gpu code.


To run the code, we would first need to download the relevant matrices from suitesparse and put them into a folder (e.g. data, physics). For instance, the location of "parabolic_fem" should be "data/parabolic_fem/parabolic_fem.mtx".
Then one should use write_graph.jl from "cpu_implementation" to write down the reordered matrices. See the jl files starting with the prefix "produce" for examples on how to create those matrices.
**Make sure to create a folder for each matrix (i.e. parabolic_fem folder would contain all variations of the parabolic_fem matrices). Folders must be manually created for matrices that are not from SuiteSparse (i.e. 3D uniform poisson, etc.). The folder must exist before running the julia script, otherwise the run might fail.**

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

