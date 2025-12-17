/**
 * Exercise 01: Device Query
 * 
 * Query and display GPU properties.
 * 
 * TODO: Complete the missing parts marked with TODO
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    int deviceCount = 0;
    
    // TODO 1: Get the number of CUDA devices
    // Hint: Use cudaGetDeviceCount()
    
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        
        // TODO 2: Get device properties
        // Hint: Use cudaGetDeviceProperties()
        
        
        printf("========================================\n");
        printf("Device %d: %s\n", dev, prop.name);
        printf("========================================\n");
        
        // Basic info
        printf("\n--- Compute Capability ---\n");
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        
        // TODO 3: Print the following properties:
        printf("\n--- Processor Info ---\n");
        // - Number of SMs (multiProcessorCount)
        // - Max threads per SM (maxThreadsPerMultiProcessor)
        // - Max threads per block (maxThreadsPerBlock)
        // - Warp size (warpSize)
        
        
        printf("\n--- Memory Info ---\n");
        // TODO 4: Print memory properties:
        // - Total global memory (totalGlobalMem) - convert to GB
        // - Shared memory per block (sharedMemPerBlock) - convert to KB
        // - Registers per block (regsPerBlock)
        // - Memory clock rate (memoryClockRate) - in MHz
        // - Memory bus width (memoryBusWidth) - in bits
        
        
        printf("\n--- Block/Grid Dimensions ---\n");
        // TODO 5: Print max block and grid dimensions
        // - maxThreadsDim[0], [1], [2] for block
        // - maxGridSize[0], [1], [2] for grid
        
        
        printf("\n--- Additional Features ---\n");
        // TODO 6: Print additional features:
        // - Concurrent kernels (concurrentKernels)
        // - Async engine count (asyncEngineCount)
        // - Unified addressing (unifiedAddressing)
        // - Managed memory (managedMemory)
        
        
        printf("\n");
    }
    
    // TODO 7 (Bonus): Calculate and print theoretical performance
    // Peak FLOPS = SMs × CUDA_cores_per_SM × Clock_rate × 2 (FMA)
    // Note: CUDA cores per SM varies by architecture
    //   - Turing (7.5): 64 FP32 cores per SM
    //   - Ampere (8.x): 64 FP32 cores per SM  
    //   - Ada (8.9): 128 FP32 cores per SM
    //   - Hopper (9.0): 128 FP32 cores per SM
    
    return 0;
}
