HIPCC = hipcc

# Compiler flags
INC = -I$(PAPI_DIR)/include -I/opt/rocm/include/rocblas -I/opt/rocm/include/rocsparse
# -I/opt/rocm/include/amd_smi
LIB = -L$(PAPI_DIR)/lib -lpapi -L/opt/rocm/lib -lrocblas -lrocsparse -lpthread -lamd_smi 
CFLAGS = -O2 -Wall

# Pattern rule: For any target, compile the corresponding .cpp file.
%: %.cpp
	$(HIPCC) $(CFLAGS) $(INC) $(LIB) -o $@ $<

# Build all executables for each .cpp file in the directory.
TARGETS := $(patsubst %.cpp, %, $(wildcard *.cpp))

all: $(TARGETS)

# Clean target: only remove the compiled executables.
clean:
	rm -f $(TARGETS)

.PHONY: all clean 