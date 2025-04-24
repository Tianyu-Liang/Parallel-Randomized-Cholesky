NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/nvcc



METIS_PATH = /pscratch/sd/t/tianyul/parmetis/local

CXXFLAGS = -std=c++20 -O2 \
           -I$(METIS_PATH)/include \
		   -I/pscratch/sd/t/tianyul/fast_matrix_market/include/ \
		   -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux/lib/

LDFLAGS = -lparmetis -lGKlib \
          -L$(METIS_PATH)/lib -lmetis \
          -lm

CUDAFLAGS = -arch=sm_80 -lcusparse -lcudart -lcublas



TARGET = driver
SRC = driver.cu
HEADERS = pre_process.hpp auxilliary.hpp solver.hpp

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS)  -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)