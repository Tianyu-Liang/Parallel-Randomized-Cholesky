NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc





CXXFLAGS = -std=c++20 -O2 \
		   -I/pscratch/sd/t/tianyul/fast_matrix_market/include/ \
		   -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib/

LDFLAGS = 

CUDAFLAGS = -arch=sm_80 -lcusparse -lcudart -lcublas



TARGET_GRAPH = driver
TARGET_PHYSICS = driver_physics
HEADERS = pre_process.hpp auxilliary.hpp solver.hpp

all: $(TARGET_GRAPH) $(TARGET_PHYSICS)

$(TARGET_GRAPH): driver.cu $(HEADERS)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) -o $(TARGET_GRAPH) driver.cu $(LDFLAGS)

$(TARGET_PHYSICS): driver_physics.cu $(HEADERS)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) -o $(TARGET_PHYSICS) driver_physics.cu $(LDFLAGS)

clean:
	rm -f $(TARGET_GRAPH) $(TARGET_PHYSICS)