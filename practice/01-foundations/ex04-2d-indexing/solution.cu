/**
 * Exercise 04: 2D Grid Indexing - SOLUTION
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

// 2D kernel that scales each matrix element
__global__ void matrixScale(float *input, float *output, 
                            float scalar, int width, int height) {
    // Calculate column (x) and row (y) from thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (col < width && row < height) {
        // Calculate 1D index from 2D coordinates (row-major)
        int index = row * width + col;
        
        // Perform the scaling operation
        output[index] = input[index] * scalar;
    }
}

// Helper function to print a small matrix
void printMatrix(float *matrix, int width, int height, const char *name) {
    printf("%s:\n", name);
    for (int row = 0; row < height && row < 8; row++) {
        printf("  ");
        for (int col = 0; col < width && col < 8; col++) {
            printf("%6.2f ", matrix[row * width + col]);
        }
        if (width > 8) printf("...");
        printf("\n");
    }
    if (height > 8) printf("  ...\n");
    printf("\n");
}

int main() {
    // Matrix dimensions
    int width = 1024;
    int height = 768;
    float scalar = 2.5f;
    
    size_t size = width * height * sizeof(float);
    
    printf("Matrix Scaling: %d x %d matrix\n", width, height);
    printf("Scalar: %.2f\n", scalar);
    printf("Total elements: %d\n", width * height);
    printf("Memory: %.2f MB\n\n", size / 1e6);
    
    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    // Initialize input matrix
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(i % 10);  // Values 0-9 repeating
    }
    
    printf("Before scaling:\n");
    printMatrix(h_input, width, height, "Input");
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Set up 2D block and grid dimensions
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    printf("Launch configuration:\n");
    printf("  Threads per block: %d x %d = %d\n", 
           threadsPerBlock.x, threadsPerBlock.y,
           threadsPerBlock.x * threadsPerBlock.y);
    printf("  Blocks in grid: %d x %d = %d\n",
           blocksPerGrid.x, blocksPerGrid.y,
           blocksPerGrid.x * blocksPerGrid.y);
    printf("  Total threads: %d\n\n",
           threadsPerBlock.x * threadsPerBlock.y * 
           blocksPerGrid.x * blocksPerGrid.y);
    
    // Launch kernel
    matrixScale<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, scalar, width, height
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    printf("After scaling:\n");
    printMatrix(h_output, width, height, "Output");
    
    // Verify results
    bool success = true;
    for (int i = 0; i < width * height; i++) {
        float expected = h_input[i] * scalar;
        if (fabsf(h_output[i] - expected) > 0.001f) {
            printf("Error at index %d: expected %.2f, got %.2f\n", 
                   i, expected, h_output[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("✅ PASSED! All elements scaled correctly.\n");
    } else {
        printf("❌ FAILED! Results are incorrect.\n");
    }
    
    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    
    return success ? 0 : 1;
}
