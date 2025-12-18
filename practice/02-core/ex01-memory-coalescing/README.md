# Exercise: Memory Coalescing

Practice identifying and fixing memory coalescing issues.

## Learning Goals
- Recognize coalesced vs strided access patterns
- Convert AoS to SoA data structures
- Measure bandwidth improvements

## Tasks

### Task 1: Fix Strided Access
The kernel below has poor coalescing. Fix it.

```cpp
// BROKEN: Strided access
__global__ void process(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * 4;  // Problem!
    if (strided_idx < n) {
        data[strided_idx] *= 2.0f;
    }
}
```

### Task 2: AoS to SoA Conversion
Convert this structure for better GPU performance:

```cpp
// Convert this:
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};
Particle particles[N];

// To SoA format and update the kernel
```

### Task 3: Benchmark
Measure bandwidth for:
- Stride-1 (coalesced)
- Stride-2, 4, 8, 16, 32

## Files
- `coalescing.cu` - Starter code
- `solution.cu` - Reference solution

## Success Criteria
- [ ] Achieve >80% of theoretical memory bandwidth with coalesced access
- [ ] AoS to SoA conversion shows >2x speedup
- [ ] Can explain why stride-32 is worst case
