# Exercise 03: Vector Addition

Your first real computation on the GPU - adding two arrays!

## Learning Goals
- Pass data to and from the GPU
- Use `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- Calculate global thread index
- Handle arrays larger than thread count

## Instructions

1. Complete `vector_add.cu`
2. Compile: `nvcc vector_add.cu -o vector_add`
3. Run: `./vector_add`

## Key Concepts

### Memory Allocation
```cpp
float *d_a;  // Device pointer
cudaMalloc(&d_a, size);  // Allocate on GPU
```

### Memory Transfer
```cpp
// Host to Device
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

// Device to Host
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
```

### Global Thread Index
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

### Bounds Checking
```cpp
if (i < n) {
    c[i] = a[i] + b[i];
}
```

## Tasks

- [ ] Allocate device memory for 3 arrays
- [ ] Copy input arrays to device
- [ ] Write kernel to add elements
- [ ] Copy result back to host
- [ ] Verify result is correct
- [ ] Free all memory

## Common Mistakes

1. **Forgetting bounds check** → Memory corruption
2. **Wrong cudaMemcpy direction** → Garbage data
3. **Not freeing memory** → Memory leaks
4. **Wrong grid/block size** → Missing elements

## Bonus

1. Time the GPU vs CPU version
2. Try different block sizes (32, 64, 128, 256, 512)
3. Handle very large arrays (10 million elements)
