#include <iostream>
#include <vector>

#include <assert.h>

// Note: 1e-6 fails.
#define MAX_ERR 1e-5

using namespace std;

__global__ void matMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.f;

    // this will only work for square matricies.
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }    
}

int main() {

    // num rows in N*N matrix
    int N = 16;
    int SIZE = N * N;

    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = sin(i);
            h_B[i * N + j] = cos(j);
        }
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **) &d_A, sizeof(float) * SIZE);
    cudaMalloc((void **) &d_B, sizeof(float) * SIZE);
    cudaMalloc((void **) &d_C, sizeof(float) * SIZE);

    // Copy host memory to device memory
    cudaMemcpy(d_A, &h_A[0], sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &h_B[0], sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    // Call cuda kernel
    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);
    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    //
    // Copy device memory to host.
    //
    // We don't need to explicitly call cudaDeviceSynchronize because
    // cudaMemcpy is a blocking operation and implcitly waits for the
    // GPU to finish all previous operations, including any kernels.
    //
    cudaMemcpy(&h_C[0], d_C, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

    // Now do the matrix multiplication on the CPU
    vector<float> cpu_C(SIZE);
    float sum;

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            sum = 0.f;

            for (int n = 0; n < N; n++) {
                // C[row][column] = A[row] * B[column]
                sum += h_A[r * N + n] * h_B[n * N + c];
            }

            cpu_C[r * N + c] = sum;
        }
    }

    // debug the checker function below
    // memcpy(&h_C[0], &cpu_C[0], sizeof(float) * SIZE);

    // Compare CPU and GPU results.
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            assert(fabs(cpu_C[r * N + c] - h_C[r * N + c]) < MAX_ERR);
        }
    }

    cout << "PASSED" << endl;

    // Deallocate device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}