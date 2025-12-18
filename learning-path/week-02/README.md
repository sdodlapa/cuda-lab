# Week 2: Memory Patterns & Optimization

## Learning Goals
- Understand coalesced memory access
- Use shared memory effectively
- Avoid bank conflicts
- Profile memory behavior

## Daily Schedule

| Day | Topic | Notebook |
|-----|-------|----------|
| 1 | Memory Coalescing | [day-1-memory-coalescing.ipynb](day-1-memory-coalescing.ipynb) |
| 2 | Shared Memory Basics | [day-2-shared-memory.ipynb](day-2-shared-memory.ipynb) |
| 3 | Bank Conflicts | [day-3-bank-conflicts.ipynb](day-3-bank-conflicts.ipynb) |
| 4 | Constant & Texture Memory | [day-4-special-memory.ipynb](day-4-special-memory.ipynb) |
| 5 | Practice & Quiz | Exercises + [checkpoint-quiz.md](checkpoint-quiz.md) |

## Key Concepts

### Memory Hierarchy (Fastest → Slowest)
1. **Registers** - Per-thread, ~0 cycles
2. **Shared Memory** - Per-block, ~5 cycles
3. **L1/L2 Cache** - Automatic, ~30-200 cycles
4. **Global Memory** - All threads, ~400+ cycles

### Memory Coalescing
When threads in a warp access consecutive memory addresses, the GPU can combine these into fewer memory transactions.

```cpp
// ✅ Coalesced: threads access consecutive addresses
data[threadIdx.x]

// ❌ Strided: poor coalescing
data[threadIdx.x * stride]
```

### Shared Memory
Fast on-chip memory shared by all threads in a block:

```cpp
__shared__ float cache[256];
cache[threadIdx.x] = global_data[idx];
__syncthreads();  // Barrier - all threads must reach here
```

### Bank Conflicts
Shared memory is divided into 32 banks. Conflicts occur when multiple threads access the same bank simultaneously.

## Project: Gaussian Blur
Implement an image blur filter using shared memory tiling to reduce global memory access.

## Resources
- [CUDA C++ Programming Guide - Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [CUDA Best Practices - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
