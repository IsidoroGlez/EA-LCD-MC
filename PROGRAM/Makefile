# Global defines
SHELL=bash
L := 16
Lz := 48
NBETAS := 40
NBITSPREBUSQUEDAS := 12
GLOBAL_DEFINES := -DL=$(L) -DLz=$(Lz) -DNBETAS=$(NBETAS) -DNUMBITSPREBUSQUEDAS=$(NBITSPREBUSQUEDAS)

OBC := 1
WEAK := 0
STRONG := 0

NO_CLUSTER := 0

ifeq ($(OBC), 0)
    GLOBAL_DEFINES += -DPBC
    BC_SUFFIX := _PBC
else
    GLOBAL_DEFINES += -DOBC
ifeq ($(WEAK), 1)
    GLOBAL_DEFINES += -DWEAK
    BC_SUFFIX := _OBC_WEAK
else ifeq ($(STRONG),1)
    GLOBAL_DEFINES += -DSTRONG
    BC_SUFFIX := _OBC_STRONG
else
    BC_SUFFIX := _OBC
endif
endif

ifeq ($(NO_CLUSTER),1)
    GLOBAL_DEFINES += -DNO_CLUSTER
    NO_CLUSTER_SUFFIX := _NO_CLUSTER
else
    NO_CLUSTER_SUFFIX :=
endif

# Directory paths
SRC_DIR := src
INCLUDE_DIR := include
OBJ_DIR := obj
BIN_DIR := bin

# Build type: release (default), debug, gdb
BUILD ?= release
# Build configuration
ifeq ($(BUILD),release)
    BUILD_DEFINES :=
    C_OPT_FLAGS := -O3
    CUDA_OPT_FLAGS := -O3
    CUDA_DEBUG_FLAGS :=
    PTXAS_FLAGS := --ptxas-options=-v

else ifeq ($(BUILD),debug)
    BUILD_DEFINES := -DDEBUG
    C_OPT_FLAGS := -O0 -g
    CUDA_OPT_FLAGS := -O0
    CUDA_DEBUG_FLAGS := -G -g -lineinfo
    PTXAS_FLAGS :=

else ifeq ($(BUILD),gdb)
    BUILD_DEFINES := -DGDB
    C_OPT_FLAGS := -O0 -g
    CUDA_OPT_FLAGS := -O0
    CUDA_DEBUG_FLAGS := -G -g -lineinfo
    PTXAS_FLAGS :=

else
    $(error BUILD invalid: $(BUILD))
endif


# CUDA compilation mode: native (default), portable, native+portable
CUDA_MODE ?= native
# CUDA_MODE parsing: allows native, portable, native+portable, portable+native
CUDA_NATIVE    := $(findstring native,$(CUDA_MODE))
CUDA_PORTABLE  := $(findstring portable,$(CUDA_MODE))

ifeq ($(CUDA_NATIVE),native)
    HOST_NATIVE_FLAGS := -march=native -mtune=native
else
    HOST_NATIVE_FLAGS :=
endif

# Compiler flags
DEPFLAGS := -MMD -MP

C_FLAGS := -c $(C_OPT_FLAGS) $(HOST_NATIVE_FLAGS) \
    $(GLOBAL_DEFINES) $(BUILD_DEFINES) \
    $(DEPFLAGS) \
    -m64 -I$(INCLUDE_DIR)

CUDADIR := /usr/local/cuda
CUDAINCLUDEDIR := $(CUDADIR)/include

# Portable CUDA architectures (fat binary)
CUDA_ARCHS_PORTABLE := \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_80,code=compute_80

ifeq ($(CUDA_NATIVE),native)
    CUDA_ARCHS += -arch=native
ifeq ($(shell nvidia-smi >/dev/null 2>&1 && echo yes),)
    $(error CUDA_MODE=native without visible device)
endif
endif

ifeq ($(CUDA_PORTABLE),portable)
    CUDA_ARCHS += $(CUDA_ARCHS_PORTABLE)
endif

ifeq ($(strip $(CUDA_ARCHS)),)
    $(error CUDA_MODE invalid: $(CUDA_MODE))
endif

ifneq ($(strip $(HOST_NATIVE_FLAGS)),)
    NVCC_HOST_FLAGS := -Xcompiler="$(HOST_NATIVE_FLAGS)"
else
    NVCC_HOST_FLAGS :=
endif

NVCC_FLAGS := -c $(CUDA_OPT_FLAGS) $(CUDA_DEBUG_FLAGS) \
    $(DEPFLAGS) \
    $(PTXAS_FLAGS) \
    $(CUDA_ARCHS) \
    $(NVCC_HOST_FLAGS) \
    $(GLOBAL_DEFINES) $(BUILD_DEFINES) \
    -I$(INCLUDE_DIR) -I$(CUDAINCLUDEDIR)

LDFLAGS := -lm -lcurand

# Compiler commands
CC := /usr/bin/g++
NVCC := nvcc

# Target executable name
TARGET := $(BIN_DIR)/EA_CUBE_L${L}_Lz${Lz}_NB${NBETAS}_NBPRE${NBITSPREBUSQUEDAS}$(BC_SUFFIX)$(NO_CLUSTER_SUFFIX)

# Source files
C_SOURCES := $(wildcard $(SRC_DIR)/*.c)
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)

# Object files
C_OBJECTS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(C_SOURCES))
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))

# All object files
OBJECTS := $(C_OBJECTS) $(CU_OBJECTS)

# Default target
all: $(TARGET)
	@echo "Build complete: $(TARGET)"

# Link the final executable
$(TARGET): $(OBJECTS)
	@echo "Creating file $@"
	@mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $@ $(LDFLAGS)

# Compile C source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) $(C_FLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu Makefile
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/*~ *~ \
	       $(SRC_DIR)/*~ $(INCLUDE_DIR)/*~ \
	       $(OBJ_DIR)/*.d
# Help
help:
	@echo ""
	@echo "Targets:"
	@echo "  make                     Build project"
	@echo "  make clean               Remove build artifacts"
	@echo "  make help                Show this help"
	@echo ""
	@echo "Build configuration (BUILD):"
	@echo "  release   (default)      Optimized build"
	@echo "  debug                    CPU + CUDA debug (-DDEBUG, -G, -lineinfo)"
	@echo "  gdb                      For kernel debugging (-DGDB)"
	@echo ""
	@echo "CUDA compilation mode (CUDA_MODE):"
	@echo "  native    (default)      Optimized for local GPU only"
	@echo "  portable                 Portable fat binary (multiple GPUs + PTX)"
	@echo "  native+portable          Native + portable fat binary"
	@echo "  portable+native          Same as above (order does not matter)"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make BUILD=debug"
	@echo "  make BUILD=gdb"
	@echo "  make CUDA_MODE=portable"
	@echo "  make CUDA_MODE=native+portable BUILD=debug"
	@echo ""

# Phony targets
.PHONY: all clean help

# Automatic dependency files
DEPS := $(OBJECTS:.o=.d)
-include $(DEPS)
