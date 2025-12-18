/**
 * Shared Memory Stencil Exercise
 * 
 * Implement a 3-point stencil using shared memory.
 * output[i] = (input[i-1] + input[i] + input[i+1]) / 3.0
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)
#define BLOCK_SIZE 256
#define RADIUS 1

// Naive version (no shared memory)
__global__ void stencilNaive(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n - 1) {
        output[idx] = (input[idx - 1] + input[idx] + input[idx + 1]) / 3.0f;
    } else if (idx < n) {
        output[idx] = input[idx];
    }
}

// TODO: Implement using shared memory
__global__ void stencilShared(float *input, float *output, int n) {
    // Hint: Allocate shared memory for BLOCK_SIZE + 2*RADIUS elements
    // __shared__ float tile[BLOCK_SIZE + 2 * RADIUS];
    
    // TODO:
    // 1. Calculate global index
    // 2. Load center elements into shared memory
    // 3. Load halo elements (threads at edges of block)
    // 4. __syncthreads()
    // 5. Compute stencil using shared memory
    // 6. Write result to output
}

int main() {
    printf("=== Shared Memory Stencil Exercise ===\n\n");
    
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Run naive version
    stencilNaive<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify a few values
    printf("Naive results:\n");
    printf("  output[1] = %.2f (expected: %.2f)\n", 
           h_output[1], (h_input[0] + h_input[1] + h_input[2]) / 3.0f);
    printf("  output[100] = %.2f (expected: %.2f)\n",
           h_output[100], (h_input[99] + h_input[100] + h_input[101]) / 3.0f);
    
    // TODO: Run and verify shared memory version
    printf("\nTODO: Implement stencilShared and benchmark!\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}
