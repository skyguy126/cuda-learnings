.PHONY : all
all : hello.out vector_add.out

hello.out: hello.cu
	nvcc --compiler-options -Wall -o hello.out hello.cu

vector_add.out: vector_add.cu
	nvcc --compiler-options -Wall -o vector_add.out vector_add.cu

clean:
	rm -f *.out
