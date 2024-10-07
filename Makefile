.PHONY : all clean

# Automatically collect all .cu source files
CU_FILES := $(wildcard *.cu)

# List of .cu files to exclude
EXCLUDE_CU := cuda_check.cu

# Automatically generate corresponding .out files by changing .cu to .out
OUT_FILES := $(patsubst %.cu, %.out, $(filter-out $(EXCLUDE_CU), $(CU_FILES)))

# The all rule depends on all .out files
all: $(OUT_FILES)

# Generic rule for compiling .cu files into .out files
%.out: %.cu
	nvcc --compiler-options -Wall -o $@ $<

clean:
	rm -f *.out
