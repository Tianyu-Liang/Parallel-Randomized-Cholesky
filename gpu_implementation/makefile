NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/nvcc





CXXFLAGS = -std=c++20 -O2 \
		   -I/pscratch/sd/t/tianyul/fast_matrix_market/include/ \
		   -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux/lib/

LDFLAGS = 

CUDAFLAGS = -arch=sm_80 -lcusparse -lcudart -lcublas



TARGET = driver
SRC = driver.cu
HEADERS = pre_process.hpp auxilliary.hpp solver.hpp

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS)  -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)