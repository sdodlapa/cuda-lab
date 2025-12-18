# Week 2 Checkpoint Quiz: Memory Patterns & Optimization

Test your understanding of GPU memory hierarchy and optimization techniques.

---

## Section 1: Memory Coalescing (Day 1)

### Q1: Coalesced Access Pattern
Which access pattern is **coalesced**?

```cpp
// Pattern A
data[threadIdx.x]

// Pattern B  
data[threadIdx.x * 2]

// Pattern C
data[blockIdx.x]
```

<details>
<summary>Answer</summary>

**Pattern A** is coalesced. Consecutive threads (0, 1, 2, ...) access consecutive memory addresses (data[0], data[1], data[2], ...).

- Pattern B has stride-2 access (skips every other element)
- Pattern C has all threads in a block reading the same address

</details>

---

### Q2: AoS vs SoA
You have particle data with position (x, y, z) and velocity (vx, vy, vz). Which structure is better for GPU coalescing?

```cpp
// Option A: Array of Structures
struct Particle { float x, y, z, vx, vy, vz; };
Particle particles[N];

// Option B: Structure of Arrays
struct Particles {
    float x[N], y[N], z[N];
    float vx[N], vy[N], vz[N];
};
```

<details>
<summary>Answer</summary>

**Option B: Structure of Arrays (SoA)** is better for GPU coalescing.

When kernel updates all x-coordinates:
- **SoA**: `particles.x[idx]` → consecutive threads access consecutive memory
- **AoS**: `particles[idx].x` → stride-6 access (each struct is 6 floats apart)

</details>

---

### Q3: Matrix Transpose Problem
In naive matrix transpose, why is the **write** problematic?

```cpp
output[col * width + row] = input[row * width + col];
```

<details>
<summary>Answer</summary>

The read `input[row * width + col]` is **coalesced** (consecutive threads read consecutive columns).

But the write `output[col * width + row]` is **strided** - consecutive threads write to addresses `width` apart, causing poor memory bandwidth utilization.

</details>

---

## Section 2: Shared Memory (Day 2)

### Q4: Shared Memory Declaration
What's wrong with this code?

```cpp
__global__ void kernel(float *data, int n) {
    __shared__ float tile[n];  // Dynamic based on parameter
    // ...
}
```

<details>
<summary>Answer</summary>

Shared memory size must be known at **compile time** for static allocation. 

Fixes:
```cpp
// Option 1: Compile-time constant
__shared__ float tile[256];

// Option 2: Dynamic allocation (size passed at launch)
extern __shared__ float tile[];
// Launch: kernel<<<blocks, threads, size_bytes>>>(...)
```

</details>

---

### Q5: __syncthreads() Requirement
Why is `__syncthreads()` necessary here?

```cpp
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Phase 1: Load
tile[ty][tx] = input[row * width + col];

// Phase 2: Use
float result = tile[ty][tx] + tile[ty][(tx + 1) % TILE_SIZE];
```

<details>
<summary>Answer</summary>

Without `__syncthreads()` between load and use phases:
- Thread 0 might try to read `tile[0][1]` before Thread 1 has written it
- **Race condition**: reading uninitialized data

Fix:
```cpp
tile[ty][tx] = input[...];
__syncthreads();  // Wait for ALL threads to finish loading
float result = tile[ty][tx] + tile[ty][(tx + 1) % TILE_SIZE];
```

</details>

---

### Q6: Shared Memory Transpose
How does shared memory fix the transpose problem?

<details>
<summary>Answer</summary>

**Strategy:**
1. **Coalesced read** from input → write to shared memory (row-major)
2. **Sync threads**
3. Read from shared memory (column-major) → **coalesced write** to output

Shared memory is ~100x faster than global memory, so the column-major read from shared memory is essentially free. Both global memory operations become coalesced.

</details>

---

## Section 3: Bank Conflicts (Day 3)

### Q7: Bank Calculation
Shared memory has **32 banks**. What bank does address `shared[65]` map to?

<details>
<summary>Answer</summary>

Bank = address % 32 = 65 % 32 = **Bank 1**

(Assuming 4-byte elements where each bank holds consecutive 4-byte words)

</details>

---

### Q8: Bank Conflict Identification
Which access has a **32-way bank conflict**?

```cpp
// Pattern A
shared[threadIdx.x]

// Pattern B
shared[threadIdx.x * 2]

// Pattern C  
shared[threadIdx.x * 32]
```

<details>
<summary>Answer</summary>

**Pattern C** has 32-way bank conflict.

- Pattern A: Each thread hits different bank (thread i → bank i % 32) ✅
- Pattern B: 2-way conflict (threads 0,16 hit bank 0; threads 1,17 hit bank 1)
- Pattern C: All 32 threads hit bank 0 (0×32=0, 1×32=32, 2×32=64... all % 32 = 0) ❌

</details>

---

### Q9: Padding Solution
How does "+1 padding" fix bank conflicts in matrix transpose?

```cpp
// Without padding
__shared__ float tile[32][32];     // Problem with column access

// With padding
__shared__ float tile[32][32 + 1]; // Fixed!
```

<details>
<summary>Answer</summary>

**Without padding:** Column access `tile[0][0], tile[1][0], tile[2][0]...` maps to:
- tile[0][0] → index 0 → bank 0
- tile[1][0] → index 32 → bank 0
- tile[2][0] → index 64 → bank 0
- All same bank! 32-way conflict.

**With +1 padding:** Each row is 33 elements:
- tile[0][0] → index 0 → bank 0
- tile[1][0] → index 33 → bank 1
- tile[2][0] → index 66 → bank 2
- Different banks! No conflict.

</details>

---

## Section 4: Constant & Texture Memory (Day 4)

### Q10: Constant Memory Use Case
When is constant memory most effective?

<details>
<summary>Answer</summary>

Constant memory is most effective when **all threads in a warp read the same value**.

**Best use cases:**
- Convolution filter coefficients
- Lookup tables accessed by all threads
- Physical constants (gravity, pi, etc.)

**Why:** Constant memory has a special cache that can **broadcast** a single value to all 32 threads in a warp in one cycle.

</details>

---

### Q11: Constant Memory API
What's the correct way to copy data to constant memory?

```cpp
__constant__ float d_filter[64];

// Option A
cudaMemcpy(d_filter, h_filter, size, cudaMemcpyHostToDevice);

// Option B
cudaMemcpyToSymbol(d_filter, h_filter, size);
```

<details>
<summary>Answer</summary>

**Option B** is correct: `cudaMemcpyToSymbol()`

Constant memory uses symbol-based API:
- `cudaMemcpyToSymbol()` - copy host → constant
- `cudaMemcpyFromSymbol()` - copy constant → host

Option A would fail because `d_filter` is a symbol, not a device pointer.

</details>

---

### Q12: Memory Type Selection
Match each scenario to the best memory type:

| Scenario | Memory Type |
|----------|-------------|
| 1. Filter coefficients (all threads read same values) | ? |
| 2. Large input array (each thread reads its own element) | ? |
| 3. Tile of data shared within a block | ? |
| 4. Per-thread accumulator variable | ? |

<details>
<summary>Answer</summary>

| Scenario | Memory Type |
|----------|-------------|
| 1. Filter coefficients (all threads read same values) | **Constant Memory** |
| 2. Large input array (each thread reads its own element) | **Global Memory** |
| 3. Tile of data shared within a block | **Shared Memory** |
| 4. Per-thread accumulator variable | **Registers** |

</details>

---

## Bonus: Code Analysis

### Q13: Optimize This Kernel
What optimizations can you apply?

```cpp
struct Point { float x, y, z; };  // AoS

__global__ void processPoints(Point *points, float *filter, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 9; i++) {
            sum += points[idx].x * filter[i];  // filter read
        }
        output[idx] = sum;
    }
}
```

<details>
<summary>Answer</summary>

**Optimizations:**

1. **Convert AoS to SoA** - avoid strided access on `points[idx].x`
```cpp
struct Points { float *x, *y, *z; };  // SoA
```

2. **Move filter to constant memory** - all threads read same filter[i]
```cpp
__constant__ float d_filter[9];
// Use cudaMemcpyToSymbol() to copy
```

3. **Consider shared memory** - if points are reused within a block

**Optimized:**
```cpp
__constant__ float d_filter[9];

__global__ void processPoints(float *x, float *y, float *z, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        float xi = x[idx];  // Coalesced read
        for (int i = 0; i < 9; i++) {
            sum += xi * d_filter[i];  // Broadcast from constant cache
        }
        output[idx] = sum;
    }
}
```

</details>

---

## Score Yourself

| Score | Level |
|-------|-------|
| 11-13 | 🏆 Memory Master - Ready for Week 3! |
| 8-10 | 👍 Good understanding - Review weak areas |
| 5-7 | 📚 Needs review - Re-read the notebooks |
| 0-4 | 🔄 Start over - Focus on one concept at a time |

---

## Week 2 Key Takeaways

1. **Coalescing** = consecutive threads → consecutive memory addresses
2. **SoA > AoS** for GPU data structures
3. **Shared memory** = programmer-managed cache for data reuse
4. **__syncthreads()** = barrier for all threads in a block
5. **32 banks** in shared memory; stride-32 = worst conflict
6. **+1 padding** eliminates column access bank conflicts
7. **Constant memory** = broadcast cache for read-only data

**Next: Week 3 - Parallel Algorithms (Reduction, Scan, Histogram)**
