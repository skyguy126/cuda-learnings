#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {

    //
    // <<<...>>> is called the kernel launch.
    // 1 block and 1 thread per block.
    //

    // Call CUDA kernel
    cuda_hello<<<1,1>>>(); 

    //
    // Need to call this if we are using printf(...) in the kernel.
    // This ensures that the CPU waits for the GPU to finish executing before proceeding.
    //

    cudaDeviceSynchronize();

    return 0;
}
