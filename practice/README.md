# CUDA Practice Exercises

Hands-on CUDA programming exercises organized by topic and difficulty.

## 🆓 Run on Google Colab (Free GPU!)

**No local GPU? No problem!** All exercises can run on Google Colab's free T4 GPU.

👉 **[Colab Setup Guide](COLAB-SETUP.md)** - Complete instructions for running exercises in your browser

### Quick Launch - Exercises with Colab Notebooks

| Exercise | Description | Colab |
|----------|-------------|-------|
| Device Query | Query GPU properties | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex01-device-query/colab-device-query.ipynb) |
| Hello GPU | First CUDA kernel | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex02-hello-gpu/colab-hello-gpu.ipynb) || 03: Vector Add | Memory transfers & computation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex03-vector-add/colab-vector-add.ipynb) |
| 04: 2D Indexing | 2D grids for matrices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex04-2d-indexing/colab-2d-indexing.ipynb) |
---

## 📂 Exercise Structure

Exercises are organized by topic and increasing complexity:

```
practice/
├── 01-foundations/          # Start here!
│   ├── ex01-device-query/   # Query GPU properties
│   └── ex02-hello-gpu/      # Your first kernel
├── 02-core/                 # Core CUDA patterns
│   ├── histogram/
│   ├── matrix/
│   ├── reduction/
│   ├── scan/
│   └── sorting/
├── 03-memory/              # Memory optimizations
├── 04-performance/         # Performance tuning
├── 05-advanced/            # Advanced features
└── 06-specialization/      # Domain-specific
```

---

## 🎯 01-Foundations (Start Here!)

### Exercise 01: Device Query
**Goal**: Query and display GPU properties

**Topics**: `cudaGetDeviceCount()`, `cudaGetDeviceProperties()`, GPU specs

📝 [README](01-foundations/ex01-device-query/README.md) | 💻 [Starter Code](01-foundations/ex01-device-query/device_query.cu) | ✅ [Solution](01-foundations/ex01-device-query/solution.cu) | 🚀 [Colab](01-foundations/ex01-device-query/colab-device-query.ipynb)

---

### Exercise 02: Hello GPU
**Goal**: Write your first CUDA kernel

**Topics**: `__global__` functions, kernel launch syntax `<<<blocks, threads>>>`, thread indexing

📝 [README](01-foundations/ex02-hello-gpu/README.md) | 💻 [Starter Code](01-foundations/ex02-hello-gpu/hello_gpu.cu) | ✅ [Solution](01-foundations/ex02-hello-gpu/solution.cu) | 🚀 [Colab](01-foundations/ex02-hello-gpu/colab-hello-gpu.ipynb)

---

### Exercise 03: Vector Addition ⭐ NEW
**Goal**: Your first real GPU computation

**Topics**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, global thread indexing, bounds checking

📝 [README](01-foundations/ex03-vector-add/README.md) | 💻 [Starter Code](01-foundations/ex03-vector-add/vector_add.cu) | ✅ [Solution](01-foundations/ex03-vector-add/solution.cu) | 🚀 [Colab](01-foundations/ex03-vector-add/colab-vector-add.ipynb)

**Tasks**:
- Allocate GPU memory
- Transfer data host ↔ device
- Launch kernel with correct grid size
- Verify results

---

### Exercise 04: 2D Grid Indexing ⭐ NEW
**Goal**: Work with 2D data (images, matrices)

**Topics**: `dim3`, 2D block/grid dimensions, row-major layout, matrix operations

📝 [README](01-foundations/ex04-2d-indexing/README.md) | 💻 [Starter Code](01-foundations/ex04-2d-indexing/grid_2d.cu) | ✅ [Solution](01-foundations/ex04-2d-indexing/solution.cu) | 🚀 [Colab](01-foundations/ex04-2d-indexing/colab-2d-indexing.ipynb)

**Tasks**:
- Use dim3 for 2D configurations
- Calculate row/col from thread indices
- Handle boundary conditions
- Process matrices in parallel

---

## 🔨 02-Core CUDA Patterns

### Matrix Operations
- Matrix addition
- Matrix multiplication (naive)
- Matrix multiplication (tiled)
- Matrix transpose

### Reduction
- Sum reduction
- Min/max finding
- Parallel reduction patterns

### Scan (Prefix Sum)
- Inclusive scan
- Exclusive scan
- Applications

### Histogram
- Atomic operations
- Privatization
- Coalesced access

### Sorting
- Bitonic sort
- Radix sort
- Merge sort

---

## 💾 03-Memory Optimization

### Projects
- Coalesced vs uncoalesced access
- Shared memory usage
- Memory bank conflicts
- Constant memory
- Texture memory

---

## ⚡ 04-Performance Optimization

### Streams
- Concurrent kernel execution
- Overlapping compute and transfer
- Multi-stream patterns

### Profiling
- Using nvprof/ncu
- Identifying bottlenecks
- Metrics analysis

### Graphs
- Creating CUDA graphs
- Performance benefits
- Use cases

### Optimization
- Occupancy tuning
- Warp-level optimizations
- Instruction-level optimization

---

## 🚀 05-Advanced Topics

### Dynamic Parallelism
- Kernels launching kernels
- Recursive algorithms

### Multi-GPU
- Peer-to-peer transfers
- Multi-GPU strategies
- Load balancing

### Warp-Level Primitives
- Warp shuffle
- Warp vote
- Warp match

### Tensor Cores
- Using wmma API
- Matrix operations
- Performance gains

---

## 🎓 06-Specialization

### Deep Learning
- Convolution
- Activation functions
- Batch normalization

### Scientific Computing
- N-body simulation
- Molecular dynamics
- Monte Carlo methods

### Computer Vision
- Image filters
- Edge detection
- Feature extraction

### Systems Programming
- Custom allocators
- Memory pools
- IPC mechanisms

---

## 🛠️ Local Setup (Alternative to Colab)

If you have a local NVIDIA GPU:

### Prerequisites
- NVIDIA GPU (Compute Capability 3.5+)
- CUDA Toolkit 11.0+
- C++ compiler (gcc/g++ or MSVC)

### Compile & Run
```bash
cd practice/01-foundations/ex01-device-query
nvcc device_query.cu -o device_query
./device_query
```

Or use the provided Makefile:
```bash
make
./device_query
make test  # Run tests if available
```

---

## 📖 Learning Tips

### For Each Exercise:
1. **Read** the README first
2. **Understand** the concepts
3. **Attempt** to complete TODOs yourself
4. **Compile and test** your code
5. **Compare** with the solution
6. **Experiment** with variations

### Debugging Tips:
- Always check for CUDA errors
- Use `nvidia-smi` to check GPU availability
- Use `cuda-memcheck` for memory errors
- Print intermediate values for debugging

### Optimization Workflow:
1. Get it working (correctness first)
2. Profile to find bottlenecks
3. Optimize hot spots
4. Measure improvement
5. Repeat

---

## 🔗 Additional Resources

- **[Learning Path](../learning-path/README.md)** - Interactive notebooks
- **[CUDA Programming Guide](../cuda-programming-guide/index.md)** - Reference docs
- **[Quick Reference](../notes/cuda-quick-reference.md)** - Cheatsheet
- **[Colab Setup Guide](COLAB-SETUP.md)** - Run without local GPU

---

## 🎯 Recommended Learning Order

1. ✅ **01-Foundations** (ex01-02) - Start here!
2. 📚 **[Learning Path Week 1](../learning-path/week-01/)** - Complement with notebooks
3. 🔨 **02-Core** - Practice fundamental patterns
4. 💾 **03-Memory** - Learn optimization techniques
5. ⚡ **04-Performance** - Advanced performance tuning
6. 🚀 **05-Advanced** - Cutting-edge features
7. 🎓 **06-Specialization** - Domain-specific applications

---

**Questions or issues?** Check the [main README](../README.md) or CUDA documentation.

**Ready to start?** Pick an exercise and click the Colab badge to launch! 🚀
