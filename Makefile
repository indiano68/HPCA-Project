cuCC = nvcc
NULL = /dev/null
ECHO_FLAG = 
ifeq ($(OS),Windows_NT)
    CL_PATH ?= "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/HostX64/x64/cl.exe"
    cuCFLAGS = -ccbin $(CL_PATH) -allow-unsupported-compiler -Xcompiler -O2 -D_WIN_ -std=c++17
    EXEC = .exe
	NULL = nul
	ECHO_FLAG = -e
else
    cuCFLAGS = -O3 -std=c++17
    EXEC = 
endif

# Directory settings
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_FILES = $(patsubst  $(SRC_DIR)/%.cu,$(BUILD_DIR)/%$(EXEC),$(wildcard $(SRC_DIR)/*.cu))

all: $(BUILD_DIR) $(BIN_FILES)

$(BIN_FILES):  $(BUILD_DIR)/%$(EXEC): $(SRC_DIR)/%.cu $(INC_DIR)/*
	@printf "\033[0;34mCompiling $@ ... \033[0m \n" 
	@$(cuCC) $< $(cuCFLAGS) -I $(INC_DIR) -o $@ 
	@printf "\033[0;32m$@ compiled! \033[0m \n" 

# Create build directories if they do not exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
    
# Clean build directories
clean:
	@rm -rf $(BUILD_DIR)

# Declare phony targets
.PHONY: all, clean

# Default target
.DEFAULT_GOAL := all
