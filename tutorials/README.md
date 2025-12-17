# CUDA Tutorials

Self-paced tutorials for learning CUDA programming from beginner to advanced.

## 📚 Tutorial Structure

Each tutorial includes:
- Conceptual explanation with diagrams
- Code examples
- Links to exercises
- Links to relevant CUDA Programming Guide sections

## 📖 Curriculum Overview

### Phase 1: Foundations (Weeks 1-3)
| # | Tutorial | Description |
|---|----------|-------------|
| 01 | [CPU vs GPU](01-foundations/01-cpu-vs-gpu.md) | Why parallel computing? |
| 02 | GPU Architecture | SMs, cores, warps explained |
| 03 | CUDA Ecosystem | Toolkit, driver, runtime |
| 04 | Development Setup | nvcc, nsight, debugging |
| 05 | Execution Model | Kernels, threads, blocks |
| 06 | Thread Hierarchy | 1D, 2D, 3D indexing |
| 07 | Kernel Launch | <<<grid, block>>> syntax |
| 08 | Synchronization Basics | __syncthreads() |
| 09 | Memory Spaces | Global, shared, local, constant |
| 10 | Memory Allocation | cudaMalloc, cudaFree |
| 11 | Data Transfer | cudaMemcpy patterns |
| 12 | Pinned Memory | cudaMallocHost benefits |

### Phase 2: Core CUDA (Weeks 4-7)
| # | Tutorial | Description |
|---|----------|-------------|
| 01 | Reduction (Naive) | Sum of array |
| 02 | Reduction (Optimized) | Sequential addressing |
| 03 | Reduction (Warp) | Warp-level primitives |
| 04 | Reduction (Atomic) | Using atomics |
| 05-08 | Scan | Prefix sum algorithms |
| 09-12 | Histogram & Sort | Parallel algorithms |
| 13-16 | Matrix Operations | GEMM, transpose |

### Phase 3-7
See [Full Curriculum](../notes/cuda-learning-curriculum.md)

## 🔗 Quick Links

- [CUDA Programming Guide](../cuda-programming-guide/index.md) - Reference documentation
- [Quick Reference Cheatsheet](../notes/cuda-quick-reference.md) - Common patterns
- [Practice Exercises](../practice/) - Hands-on coding

---

[🏠 Back to Main](../README.md)
