#NVCC = /global/common/software/nersc9/intel/oneapi/compiler/2024.1/bin/icpx
NVCC = /usr/bin/g++
#NVCC = /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx
#NVCC = /opt/AMD/aocc-compiler-3.2.0.sles15/bin/clang++
#CUDAPATH = /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64




CXXFLAGS =  -std=c++20 -O3 \
		   -I/pscratch/sd/t/tianyul/fast_matrix_market/include/

LDFLAGS = -fopenmp \
          -L  -lm -lgomp -lpthread 
		  
#-L$(CUDAPATH)
# -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread 
# -lgomp 
#CUDAFLAGS = -arch=sm_80 -lcusparse -lcudart -lcublas 



TARGET = driver
SRC = driver_local.cpp
HEADERS = pre_process.hpp auxilliary.hpp

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC)  $(CXXFLAGS)  -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)