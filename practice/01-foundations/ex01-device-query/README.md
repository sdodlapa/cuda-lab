# Exercise 01: Device Query

Query and display GPU properties to understand your hardware.

## Learning Goals
- Use cudaGetDeviceCount() and cudaGetDeviceProperties()
- Understand key GPU specifications
- First successful CUDA program compilation

## Instructions

1. Compile: `nvcc device_query.cu -o device_query`
2. Run: `./device_query`
3. Compare your output with NVIDIA specs for your GPU

## Key Properties to Understand

| Property | What it means |
|----------|---------------|
| Compute Capability | GPU architecture version |
| SMs (Multiprocessors) | Parallel processing units |
| CUDA Cores | Processing elements per SM |
| Global Memory | Total GPU memory |
| Shared Memory/Block | Fast on-chip memory per block |
| Max Threads/Block | Launch configuration limit |
| Warp Size | Threads executed together (32) |
| Memory Bandwidth | Data transfer rate |

## Tasks

- [ ] Complete device_query.cu
- [ ] Run and understand output
- [ ] Calculate theoretical peak FLOPS
- [ ] Compare with CPU specs

## Bonus
Calculate theoretical performance:
- Peak FLOPS = CUDA Cores × Clock Speed × 2 (FMA)
- Memory Bandwidth from specs

## Solution

See `solution.cu` after attempting yourself.
