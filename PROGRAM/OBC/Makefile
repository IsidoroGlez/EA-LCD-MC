# Global defines
SHELL=bash
L := 8
Lz := 32
NBETAS := 14
NBITSPREBUSQUEDAS := 12
ANTONIO := 0
FORTISSIMO := 0
NO_HOUDAYER := 0
GLOBAL_DEFINES := -DL=$(L) -DLz=$(Lz) -DNBETAS=$(NBETAS) -DNUMBITSPREBUSQUEDAS=$(NBITSPREBUSQUEDAS) -DANTONIO=$(ANTONIO) -DFORTISSIMO=$(FORTISSIMO)

ifeq ($(NO_HOUDAYER),1)
	GLOBAL_DEFINES += -DNO_HOUDAYER
	NO_HOUDAYER_SUFFIX := _NO_HOUDAYER
else
	NO_HOUDAYER_SUFFIX :=
endif

# Directory paths
SRC_DIR := src
INCLUDE_DIR := include
OBJ_DIR := obj
BIN_DIR := bin
#CLUSTER_DIR := Cluster  # For Houdayer clustering

# Compiler flags
C_FLAGS := -c -O3 $(GLOBAL_DEFINES) -m64 #MMD and MP for the dependencies #-mcmodel=medium
CUDADIR := /usr/local/cuda
CUDAINCLUDEDIR := $(CUDADIR)/include
NVCC_FLAGS := -c -O3 --ptxas-options=-v -DHOSTREDU -arch=sm_80 $(GLOBAL_DEFINES) -I$(CUDAINCLUDEDIR) -L$(CUDADIR)/lib -lcudart #-Xcompiler -mcmodel=medium
LDFLAGS := -lm -lcurand

# Compiler commands
CC := /usr/bin/g++
NVCC := nvcc

# Target executable name
ifeq ($(ANTONIO),1)
	TARGET := $(BIN_DIR)/EA_CUBE_L${L}_Lz${Lz}_NB${NBETAS}_NBPRE${NBITSPREBUSQUEDAS}_OBC_ANTONIO$(NO_HOUDAYER_SUFFIX)
else ifeq ($(FORTISSIMO),1)
	TARGET := $(BIN_DIR)/EA_CUBE_L${L}_Lz${Lz}_NB${NBETAS}_NBPRE${NBITSPREBUSQUEDAS}_OBC_FORTISSIMO$(NO_HOUDAYER_SUFFIX)
else
	TARGET := $(BIN_DIR)/EA_CUBE_L${L}_Lz${Lz}_NB${NBETAS}_NBPRE${NBITSPREBUSQUEDAS}_OBC_GIORGIO$(NO_HOUDAYER_SUFFIX)
endif

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
