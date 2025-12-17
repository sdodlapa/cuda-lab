/**
 * Exercise 01: Device Query - SOLUTION
 * 
 * Complete GPU device query implementation.
 */

#include <stdio.h>
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

// Helper function to get CUDA cores per SM based on architecture
int getCudaCoresPerSM(int major, int minor) {
    switch ((major << 4) + minor) {
        case 0x60: return 64;   // Pascal GP100
        case 0x61: return 128;  // Pascal GP10x
        case 0x62: return 128;  // Pascal GP10x (Tegra)
        case 0x70: return 64;   // Volta
        case 0x72: return 64;   // Volta (Tegra)
        case 0x75: return 64;   // Turing
        case 0x80: return 64;   // Ampere GA100
        case 0x86: return 128;  // Ampere GA10x
        case 0x87: return 128;  // Ampere (Tegra)
        case 0x89: return 128;  // Ada Lovelace
        case 0x90: return 128;  // Hopper
        default:   return 64;   // Unknown, assume 64
    }
}

int main() {
    int deviceCount = 0;
    
    // Get the number of CUDA devices
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        
        // Get device properties
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        
        printf("========================================\n");
        printf("Device %d: %s\n", dev, prop.name);
        printf("========================================\n");
        
        // Compute Capability
        printf("\n--- Compute Capability ---\n");
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        
        // Processor Info
        printf("\n--- Processor Info ---\n");
        int coresPerSM = getCudaCoresPerSM(prop.major, prop.minor);
        int totalCores = prop.multiProcessorCount * coresPerSM;
        printf("Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
        printf("CUDA Cores per SM: %d\n", coresPerSM);
        printf("Total CUDA Cores: %d\n", totalCores);
        printf("GPU Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
        printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Warp Size: %d\n", prop.warpSize);
        
        // Memory Info
        printf("\n--- Memory Info ---\n");
        printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("Shared Memory per SM: %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
        printf("L2 Cache Size: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
        printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        
        // Calculate memory bandwidth
        double memBandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
        printf("Theoretical Memory Bandwidth: %.2f GB/s\n", memBandwidth);
        
        // Block/Grid Dimensions
        printf("\n--- Block/Grid Dimensions ---\n");
        printf("Max Block Dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Dimensions: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        // Additional Features
        printf("\n--- Additional Features ---\n");
        printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("Async Engine Count: %d\n", prop.asyncEngineCount);
        printf("Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("Managed Memory: %s\n", prop.managedMemory ? "Yes" : "No");
        printf("Cooperative Launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");
        printf("Compute Preemption: %s\n", prop.computePreemptionSupported ? "Yes" : "No");
        
        // Calculate theoretical peak performance
        printf("\n--- Theoretical Peak Performance ---\n");
        // FP32: CUDA cores × clock × 2 (FMA)
        double peakFP32 = totalCores * prop.clockRate * 2 / 1e9;  // TFLOPS
        printf("Peak FP32 Performance: %.2f TFLOPS\n", peakFP32);
        
        // FP16: Usually 2x FP32 on modern GPUs with Tensor Cores
        if (prop.major >= 7) {
            printf("Peak FP16 Performance: ~%.2f TFLOPS (estimated)\n", peakFP32 * 2);
        }
        
        printf("\n");
    }
    
    // Driver and runtime versions
    int driverVersion, runtimeVersion;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    
    printf("========================================\n");
    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("========================================\n");
    
    return 0;
}
