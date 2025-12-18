# Exercise: Bank Conflicts

Practice identifying and eliminating shared memory bank conflicts.

## Learning Goals
- Understand the 32-bank shared memory architecture
- Identify access patterns that cause conflicts
- Apply padding to eliminate conflicts

## Background

Shared memory is divided into 32 banks. Consecutive 4-byte words go to consecutive banks:
- Address 0-3 → Bank 0
- Address 4-7 → Bank 1
- ...
- Address 124-127 → Bank 31
- Address 128-131 → Bank 0 (wraps)

**Bank conflict:** Multiple threads in a warp access different addresses in the same bank.

## Tasks

### Task 1: Identify Conflicts
For each access pattern, determine the conflict level (none, 2-way, 4-way, 32-way):

```cpp
__shared__ float data[256];

// Pattern A
data[threadIdx.x]

// Pattern B
data[threadIdx.x * 2]

// Pattern C
data[threadIdx.x * 32]

// Pattern D
data[threadIdx.x * 33]
```

### Task 2: Fix Matrix Transpose
The naive transpose has 32-way bank conflicts on column access:

```cpp
__shared__ float tile[32][32];  // Problem!

// Reading columns: tile[0][col], tile[1][col], ... all hit same bank
```

Fix it using +1 padding.

### Task 3: Benchmark
Measure performance difference between:
- No conflicts (stride-1)
- 32-way conflicts (stride-32)
- Fixed with padding

## Files
- `bank_conflicts.cu` - Starter code with benchmarks

## Success Criteria
- [ ] Correctly identify conflict levels for all patterns
- [ ] Transpose with padding >1.5x faster than without
- [ ] Can explain why stride-33 has no conflicts
