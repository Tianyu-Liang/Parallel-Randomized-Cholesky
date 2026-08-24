CXX ?= g++
FMM_INCLUDE ?=

CPPFLAGS += $(if $(FMM_INCLUDE),-I$(FMM_INCLUDE))
CXXFLAGS ?= -std=c++20 -O3
LDFLAGS += -fopenmp
LDLIBS += -lm -lgomp -lpthread

TARGET = driver
SRC = driver_local.cpp
HEADERS = pre_process.hpp auxilliary.hpp

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
