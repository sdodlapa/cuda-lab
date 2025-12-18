/**
 * Memory Coalescing Exercise
 * 
 * TODO: Fix the strided access pattern and measure improvement.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 22)
#define BLOCK_SIZE 256

// TODO: Fix this kernel - it has strided access
__global__ void processStrided(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * 4;  // This causes poor coalescing!
    if (strided_idx < n) {
        output[strided_idx] = input[strided_idx] * 2.0f;
    }
}

// TODO: Implement coalesced version
__global__ void processCoalesced(float *input, float *output, int n) {
    // Your code here
}

// Benchmarking helper
float benchmark(void (*kernel)(float*, float*, int), 
                float *d_in, float *d_out, int n, const char *name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Warmup
    kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100;
    
    float bandwidth = (2.0f * n * sizeof(float)) / (ms * 1e6);
    printf("%s: %.3f ms, %.2f GB/s\n", name, ms, bandwidth);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms;
}

int main() {
    printf("=== Memory Coalescing Exercise ===\n\n");
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // TODO: Benchmark both versions and compare
    // benchmark(processStrided, d_input, d_output, N, "Strided");
    // benchmark(processCoalesced, d_input, d_output, N, "Coalesced");
    
    printf("\nTODO: Implement processCoalesced and compare!\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
