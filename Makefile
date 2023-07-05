# Input Names
BIN_DIR = bin
SRC_DIR = src

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_GENCODES = -gencode arch=compute_50,code=sm_50 \
				-gencode arch=compute_52,code=sm_52 \
				-gencode arch=compute_60,code=sm_60 \
				-gencode arch=compute_61,code=sm_61 \
				-gencode arch=compute_61,code=compute_61

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# FFTW build
ABS_PATH = $(shell pwd)
FFTW_PATH = $(ABS_PATH)/lib/fftw-build
FFTW_LIB = -L${FFTW_PATH}/lib -lfftw3
FFTW_INCLUDE = -I${FFTW_PATH}/include 

# CUDA stuff
CUDA_LIB = -L$(CUDA_LIB_PATH) -lcudart -lcufft
CUDA_INCLUDE = -I$(CUDA_INC_PATH)

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++11 -pthread
INCLUDE = $(FFTW_INCLUDE) $(CUDA_INCLUDE)
LIBS = $(FFTW_LIB) $(CUDA_LIB)

# ------------------------------------------------------------------------------
# Make Rules
# ------------------------------------------------------------------------------
all: cpu_demo gpu_demo compare_sol

GPU_FILES = gpu_demo.cpp example_fields.cpp solution_io.cpp gpu_heat_equation.cu
CPU_FILES = cpu_demo.cpp example_fields.cpp solution_io.cpp cpu_heat_equation.cpp
COMPARE_FILES = compare_sol.cpp solution_io.cpp

compare_sol: $(addsuffix .o, $(COMPARE_FILES))
	$(GPP) $(addsuffix .o, $(addprefix $(BIN_DIR)/, $(COMPARE_FILES))) -o $(BIN_DIR)/$@ $(FLAGS) $(INCLUDE) $(LIBS)

cpu_demo: $(addsuffix .o, $(CPU_FILES))
	$(GPP) $(addsuffix .o, $(addprefix $(BIN_DIR)/, $(CPU_FILES))) -o $(BIN_DIR)/$@ $(FLAGS) $(INCLUDE) $(LIBS)

gpu_demo: $(addsuffix .o, $(GPU_FILES))
	$(NVCC) $(addsuffix .o, $(addprefix $(BIN_DIR)/, $(GPU_FILES))) -o $(BIN_DIR)/$@ $(INCLUDE) $(LIBS)

%.cpp.o: $(SRC_DIR)/%.cpp
	$(GPP) $< -c -o $(BIN_DIR)/$@ $(FLAGS) $(INCLUDE) $(LIBS)

%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $< -c -o $(BIN_DIR)/$@ $(NVCC_FLAGS) $(NVCC_GENCODES) $(INCLUDE) $(LIBS)

# Clean everything
clean:
	rm -f bin/*.o bin/*.bin bin/*.gif bin/cpu_demo bin/gpu_demo bin/compare_sol

.PHONY: clean
