NVCC ?= nvcc
FMM_INCLUDE ?=
CUDA_LIB_DIR ?=
GPU_ARCH ?= sm_80

CPPFLAGS += $(if $(FMM_INCLUDE),-I$(FMM_INCLUDE))
CXXFLAGS ?= -std=c++20 -O2
CUDAFLAGS += -arch=$(GPU_ARCH)
LDFLAGS += $(if $(CUDA_LIB_DIR),-L$(CUDA_LIB_DIR))
LDLIBS += -lcusparse -lcudart -lcublas

TARGET_GRAPH = driver
TARGET_PHYSICS = driver_physics
HEADERS = pre_process.hpp auxilliary.hpp solver.hpp

all: $(TARGET_GRAPH) $(TARGET_PHYSICS)

$(TARGET_GRAPH): driver.cu $(HEADERS)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $(CXXFLAGS) -o $(TARGET_GRAPH) driver.cu $(LDFLAGS) $(LDLIBS)

$(TARGET_PHYSICS): driver_physics.cu $(HEADERS)
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) $(CXXFLAGS) -o $(TARGET_PHYSICS) driver_physics.cu $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET_GRAPH) $(TARGET_PHYSICS)
