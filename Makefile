# Determine the operating system
  
ifeq ($(OS),Windows_NT)
    # Compiler settings for Windows
    CL_PATH ?= "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/HostX64/x64/cl.exe"
    CC = g++
    CFLAGS = -O3 -Wall -Wextra -Werror -fopenmp -std=c++17
    cuCC = nvcc
    cuCFLAGS = -ccbin $(CL_PATH) -allow-unsupported-compiler -Xcompiler -O2 -D_WIN_ 
    DEBUG_CFLAGS = -O0 -g -Wall -Wextra -Werror -fopenmp
    EXEC = .exe
else
    # Compiler settings for Unix-like systems
    CC = nvc++
    CFLAGS = -Ofast -fopenmp 
    cuCC = nvc++
    cuCFLAGS = -Ofast
    DEBUG_CFLAGS = -O0 -g
    DEBUG_CUFLAGS = -G -O0 -g
    EXEC =
endif

# Directory settings
SRC_DIR = src
BUILD_DIR = build

# Targets for CPU and CUDA builds
# cpu: $(BUILD_DIR) $(BUILD_DIR)/executable$(EXEC)
cuda: $(BUILD_DIR) $(BUILD_DIR)/executable_cuda$(EXEC)
# debug: $(BUILD_DIR) $(BUILD_DIR)/executable_debug$(EXEC)
# cuda_debug: $(BUILD_DIR) $(BUILD_DIR)/executable_debug_cuda$(EXEC)

# Create build directories if they do not exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
    
# Compile debug CUDA executable
$(BUILD_DIR)/executable_cuda$(EXEC): $(SRC_DIR)/main.cu
	@$(cuCC) $(cuCFLAGS) -o $(BUILD_DIR)/executable_cuda$(EXEC) $(SRC_DIR)/main.cu 

# Clean build directories
clean:
	@rm -rf $(BUILD_DIR)

# Declare phony targets
.PHONY: clean

# Default target
.DEFAULT_GOAL := cuda

# Specify C++ standard
CFLAGS += -std=c++17
DEBUG_CFLAGS += -std=c++17
