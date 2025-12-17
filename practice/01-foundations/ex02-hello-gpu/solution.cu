/**
 * Exercise 02: Hello GPU - SOLUTION
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from block %d, thread %d (global ID: %d)\n", 
           blockIdx.x, threadIdx.x, globalId);
}

int main() {
    printf("=== Launching with 1 block, 8 threads ===\n");
    helloKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Launching with 2 blocks, 4 threads each ===\n");
    helloKernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Launching with 4 blocks, 2 threads each ===\n");
    helloKernel<<<4, 2>>>();
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\nDone!\n");
    return 0;
}
