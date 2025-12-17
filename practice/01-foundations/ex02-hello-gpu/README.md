# Exercise 02: Hello GPU

Your first CUDA kernel - printing from the GPU!

## Learning Goals
- Write a `__global__` function (kernel)
- Launch a kernel with `<<<blocks, threads>>>`
- Understand thread/block indexing

## Instructions

1. Complete `hello_gpu.cu`
2. Compile: `nvcc hello_gpu.cu -o hello_gpu`
3. Run: `./hello_gpu`

## Key Concepts

### Kernel Declaration
```cpp
__global__ void myKernel() {
    // This runs on GPU
}
```

### Kernel Launch
```cpp
myKernel<<<numBlocks, threadsPerBlock>>>();
cudaDeviceSynchronize();  // Wait for GPU to finish
```

### Thread Identification
```cpp
int threadId = threadIdx.x;  // Thread index within block (0 to blockDim.x-1)
int blockId = blockIdx.x;    // Block index within grid (0 to gridDim.x-1)
int globalId = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
```

## Tasks

- [ ] Write a kernel that prints "Hello from thread X"
- [ ] Launch with 1 block, 8 threads
- [ ] Launch with 2 blocks, 4 threads each
- [ ] Observe the output order (may not be sequential!)

## Important Notes

1. **printf from GPU** requires compute capability 2.0+
2. **Output order is non-deterministic** - threads run in parallel
3. **Always call cudaDeviceSynchronize()** to see printf output
