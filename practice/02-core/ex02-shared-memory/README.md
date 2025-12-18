# Exercise: Shared Memory

Practice using shared memory for data reuse and synchronization.

## Learning Goals
- Declare static and dynamic shared memory
- Use `__syncthreads()` correctly
- Implement tiled algorithms

## Tasks

### Task 1: Stencil with Shared Memory
Implement a 1D stencil (3-point average) using shared memory:

```
output[i] = (input[i-1] + input[i] + input[i+1]) / 3.0
```

Requirements:
- Load tile + halo into shared memory
- Use `__syncthreads()` before computing
- Handle boundary conditions

### Task 2: Matrix Transpose
Implement matrix transpose using shared memory tiles:
- Read tile from input (coalesced)
- Write to shared memory
- Sync threads
- Read from shared memory (transposed)
- Write to output (coalesced)

### Task 3: Tiled Matrix Multiply
Implement TILE_SIZE × TILE_SIZE tiled matrix multiplication:
- Load A and B tiles into shared memory
- Compute partial products
- Accumulate across tiles

## Files
- `stencil.cu` - 1D stencil starter
- `transpose.cu` - Matrix transpose starter
- `matmul.cu` - Tiled matmul starter

## Success Criteria
- [ ] Stencil runs correctly with shared memory
- [ ] Transpose achieves >2x speedup over naive
- [ ] Tiled matmul significantly faster than naive
- [ ] No race conditions (proper `__syncthreads()` placement)
