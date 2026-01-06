# Global defines
SHELL=bash
L := 8
Lz := 32
NBETAS := 14
NBITSPREBUSQUEDAS := 12
GLOBAL_DEFINES := -DL=$(L) -DLz=$(Lz) -DNBETAS=$(NBETAS) -DNUMBITSPREBUSQUEDAS=$(NBITSPREBUSQUEDAS)

OBC := 1
WEAK := 0
STRONG := 0

NO_CLUSTER :=0

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

# Compiler flags
C_FLAGS := -c -O3 $(GLOBAL_DEFINES) -m64
CUDADIR := /usr/local/cuda
CUDAINCLUDEDIR := $(CUDADIR)/include
NVCC_FLAGS := -c -O3 --ptxas-options=-v -DHOSTREDU -arch=sm_80 $(GLOBAL_DEFINES) -I$(CUDAINCLUDEDIR) -L$(CUDADIR)/lib -lcudart
LDFLAGS := -lm -lcurand

# Compiler commands
CC := /usr/bin/g++
NVCC := nvcc

# Target executable name
TARGET := $(BIN_DIR)/EA_CUBE_L${L}_Lz${Lz}_NB${NBETAS}_NBPRE${NBITSPREBUSQUEDAS)$(BC_SUFFIX)$(NO_CLUSTER_SUFFIX)	

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
	@echo "Compilation $(TARGET)"

# Link the final executable
$(TARGET): $(OBJECTS)
	@echo "Creating file $@"
	@mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $@ $(LDFLAGS)

# Compile C source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(CC) $(C_FLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) Makefile
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)/*~ *~ $(SRC_DIR)/*~ $(INCLUDE_DIR)/*~

# Phony targets
.PHONY: all clean

