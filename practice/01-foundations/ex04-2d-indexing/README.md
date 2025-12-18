# Exercise 04: 2D Grid Indexing

Work with 2D data like images and matrices using 2D thread grids.

## Learning Goals
- Use 2D block and grid dimensions with `dim3`
- Calculate 2D thread indices
- Map thread coordinates to array indices
- Understand row-major memory layout

## Instructions

1. Complete `grid_2d.cu`
2. Compile: `nvcc grid_2d.cu -o grid_2d`
3. Run: `./grid_2d`

## Key Concepts

### 2D Launch Configuration
```cpp
dim3 threadsPerBlock(16, 16);    // 16x16 = 256 threads per block
dim3 blocksPerGrid(
    (width + 15) / 16,           // Ceiling division
    (height + 15) / 16
);
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

### 2D Thread Indexing
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;  // X = column
int row = blockIdx.y * blockDim.y + threadIdx.y;  // Y = row
```

### Row-Major Memory Layout
For a 2D array stored in 1D memory:
```cpp
int index = row * width + col;
```

```
Array[3][4] in memory:
Row 0: [0,0] [0,1] [0,2] [0,3]
Row 1: [1,0] [1,1] [1,2] [1,3]  
Row 2: [2,0] [2,1] [2,2] [2,3]

1D index: 0  1  2  3  4  5  6  7  8  9  10 11
```

## Tasks

- [ ] Create a 2D kernel that processes a matrix
- [ ] Use dim3 for block and grid dimensions
- [ ] Calculate row and column from thread indices
- [ ] Handle edge cases (partial blocks at boundaries)
- [ ] Implement matrix scaling: B[i][j] = A[i][j] * scalar

## Visualization

```
Grid of Blocks (4x3):
┌─────┬─────┬─────┬─────┐
│(0,0)│(1,0)│(2,0)│(3,0)│
├─────┼─────┼─────┼─────┤
│(0,1)│(1,1)│(2,1)│(3,1)│
├─────┼─────┼─────┼─────┤
│(0,2)│(1,2)│(2,2)│(3,2)│
└─────┴─────┴─────┴─────┘

Each Block (16x16 threads):
┌────────────────────────┐
│Thread (0,0)  ... (15,0)│
│   ...              ... │
│Thread (0,15) ...(15,15)│
└────────────────────────┘
```

## Common Mistakes

1. **Swapping x and y** → Transposed output
2. **Forgetting boundary check** → Out of bounds access
3. **Wrong index formula** → Column-major instead of row-major
4. **Block dimensions too large** → Exceeds 1024 thread limit

## Bonus

1. Implement matrix transpose
2. Process an image (grayscale inversion)
3. Apply a simple 3x3 filter
