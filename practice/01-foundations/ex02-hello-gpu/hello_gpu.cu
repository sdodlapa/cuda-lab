/**
 * Exercise 02: Hello GPU
 * 
 * Your first CUDA kernel!
 * 
 * TODO: Complete the kernel and launch configurations
 */

#include <stdio.h>
#include <cuda_runtime.h>

// TODO 1: Write a kernel that prints a greeting from each thread
// The kernel should print:
// "Hello from block X, thread Y (global ID: Z)"
// 
// Hints:
// - Use printf() - it works on GPU!
// - blockIdx.x gives block index
// - threadIdx.x gives thread index within block
// - Global ID = blockIdx.x * blockDim.x + threadIdx.x

__global__ void helloKernel() {
    // Your code here
    
}

int main() {
    printf("=== Launching with 1 block, 8 threads ===\n");
    // TODO 2: Launch helloKernel with 1 block and 8 threads
    
    
    // Don't forget to synchronize!
    cudaDeviceSynchronize();
    
    printf("\n=== Launching with 2 blocks, 4 threads each ===\n");
    // TODO 3: Launch helloKernel with 2 blocks and 4 threads per block
    
    
    cudaDeviceSynchronize();
    
    printf("\n=== Launching with 4 blocks, 2 threads each ===\n");
    // TODO 4: Launch with 4 blocks, 2 threads each
    
    
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\nDone!\n");
    return 0;
}
