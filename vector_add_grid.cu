#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int thread_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x;

    for(int i = thread_index; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    //
    // CPU memory is called host memory.
    // GPU memory is called device memory.
    //

    //
    // Common workflow in most programs.
    // 1. Allocate host memory and init host data.
    // 2. Allocate device memory.
    // 3. Transfer input data from host to device memory.
    // 4. Execute kernels.
    // 5. Copy output data from device to host memory.
    //

    // Allocate host memory
    a   = (float *) malloc(sizeof(float) * N);
    b   = (float *) malloc(sizeof(float) * N);
    out = (float *) malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void **) &d_a, sizeof(float) * N);
    cudaMalloc((void **) &d_b, sizeof(float) * N);
    cudaMalloc((void **) &d_out, sizeof(float) * N);

    // Copy host memory to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    //
    // We need the -1 here because for example:
    // Assume N = 512 and block_size = 128. Without
    // the -1 the calculation would be (512 + 128) / 128
    // which yields 5 and is wrong. We only need 4 threads
    // to map out the entire dataset. If we use -1 the
    // calculation instead becomes (512 + 128 - 1) / 128
    // which correctly produces 4.
    //

    int block_size = 256;
    int grid_size = ((N + block_size - 1) / block_size);

    // Call kernel.
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);

    //
    // Copy device memory to host.
    //
    // We don't need to explicitly call cudaDeviceSynchronize because
    // cudaMemcpy is a blocking operation and implcitly waits for the
    // GPU to finish all previous operations, including any kernels.
    //
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
