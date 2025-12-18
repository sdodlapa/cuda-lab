/**
 * Bank Conflicts Exercise
 * 
 * Identify and fix shared memory bank conflicts.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define ITERATIONS 1000

// Matrix transpose with bank conflicts
__global__ void transposeConflicts(float *output, float *input, int width) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];  // 32-way conflicts on column access!
    
    int xIdx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int yIdx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int idx = yIdx * width + xIdx;
    
    // Load row (coalesced)
    if (xIdx < width && yIdx < width) {
        tile[threadIdx.y][threadIdx.x] = input[idx];
    }
    
    __syncthreads();
    
    // Store column (bank conflicts!)
    xIdx = blockIdx.y * TILE_SIZE + threadIdx.x;
    yIdx = blockIdx.x * TILE_SIZE + threadIdx.y;
    idx = yIdx * width + xIdx;
    
    if (xIdx < width && yIdx < width) {
        output[idx] = tile[threadIdx.x][threadIdx.y];  // Column access = conflicts
    }
}

// TODO: Fix using +1 padding
__global__ void transposeNoPadding(float *output, float *input, int width) {
    // TODO: Change tile declaration to eliminate bank conflicts
    // __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Then use same logic as above
}

int main() {
    printf("=== Bank Conflicts Exercise ===\n\n");
    
    int width = 1024;
    size_t size = width * width * sizeof(float);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(width / TILE_SIZE, width / TILE_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark with conflicts
    transposeConflicts<<<grid, block>>>(d_output, d_input, width);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        transposeConflicts<<<grid, block>>>(d_output, d_input, width);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float conflicts_time;
    cudaEventElapsedTime(&conflicts_time, start, stop);
    conflicts_time /= ITERATIONS;
    
    printf("With bank conflicts: %.4f ms\n", conflicts_time);
    
    // TODO: Benchmark without conflicts (after implementing transposeNoPadding)
    printf("\nTODO: Implement transposeNoPadding with +1 padding!\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
