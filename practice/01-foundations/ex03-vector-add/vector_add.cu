/**
 * Exercise 03: Vector Addition
 * 
 * Add two vectors on the GPU.
 * 
 * TODO: Complete the missing parts marked with TODO
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// TODO 1: Write the vector addition kernel
// Each thread should add one element: c[i] = a[i] + b[i]
// Don't forget bounds checking!
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread index
    
    // Check bounds and perform addition
    
}

int main() {
    int n = 1000000;  // 1 million elements
    size_t size = n * sizeof(float);
    
    printf("Vector Addition: %d elements\n", n);
    printf("Memory per array: %.2f MB\n", size / 1e6);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // Initialize input arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // TODO 2: Allocate device memory
    float *d_a, *d_b, *d_c;
    // Use cudaMalloc for each array
    
    
    // TODO 3: Copy input data to device
    // Use cudaMemcpy with cudaMemcpyHostToDevice
    
    
    // TODO 4: Launch kernel
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = 0;  // TODO: Calculate this!
    
    printf("Launching kernel: %d blocks, %d threads/block\n", 
           blocksPerGrid, threadsPerBlock);
    
    // Launch the kernel
    
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // TODO 5: Copy result back to host
    // Use cudaMemcpy with cudaMemcpyDeviceToHost
    
    
    // Verify result
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at index %d: expected 3.0, got %.2f\n", i, h_c[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("✅ PASSED! All elements are correct.\n");
    } else {
        printf("❌ FAILED! Results are incorrect.\n");
    }
    
    // TODO 6: Free device memory
    // Use cudaFree for each array
    
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return success ? 0 : 1;
}
