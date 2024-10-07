#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++) {
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

    // Main function
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);

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
