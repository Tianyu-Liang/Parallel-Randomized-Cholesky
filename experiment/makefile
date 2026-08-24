CXX ?= g++
FMM_INCLUDE ?=
MKL_INCLUDE ?=
MKL_LIB_DIR ?=

CPPFLAGS += $(if $(FMM_INCLUDE),-I$(FMM_INCLUDE)) $(if $(MKL_INCLUDE),-I$(MKL_INCLUDE))
CXXFLAGS ?= -std=c++20 -O3
LDFLAGS += -fopenmp $(if $(MKL_LIB_DIR),-L$(MKL_LIB_DIR))
LDLIBS += -lm -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

TARGET = driver
SRC = driver_local.cpp
HEADERS = pre_process.hpp auxilliary.hpp custom_cg.hpp

TEST = independent_cg
TEST_SRC = independent_cg.cpp
TEST_HEADERS = pre_process.hpp

all: $(TARGET) $(TEST)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS) $(LDLIBS)

$(TEST): $(TEST_SRC) $(TEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(TEST) $(TEST_SRC) $(LDFLAGS) $(LDLIBS)


clean:
	rm -f $(TARGET) $(TEST)
