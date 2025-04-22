HIPCC = hipcc
PAPI ?= /storage/users/dwoun/apps/papi

# Compiler flags
INC = -I${PAPI}/include 
# -I/opt/rocm/include/amd_smi
LIB = -L${PAPI}/lib  -lpapi -L/opt/rocm/lib -lpthread
#-lrocblas -lrocsparse -lpthread -lamd_smi 
CFLAGS = -O2 -Wall

# Pattern rule: For any target, compile the corresponding .cpp file.
%: %.cpp
	$(HIPCC) $(CFLAGS) $(INC) $(LIB) -Wl,-rpath,$(PAPI)/lib -o $@ $<

# Build all executables for each .cpp file in the directory.
TARGETS := $(patsubst %.cpp, %, $(wildcard *.cpp))

all: $(TARGETS)

# Clean target: only remove the compiled executables.
clean:
	rm -f $(TARGETS)

.PHONY: all clean 