# Running CUDA Practice Exercises on Google Colab

This guide shows you how to run the CUDA practice exercises on Google Colab's free T4 GPU.

## 🚀 Quick Start

### Method 1: Use Pre-Made Colab Notebooks

Click the links below to open exercises directly in Colab:

| Exercise | Open in Colab |
|----------|---------------|
| 01: Device Query | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex01-device-query/colab-device-query.ipynb) |
| 02: Hello GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex02-hello-gpu/colab-hello-gpu.ipynb) |

### Method 2: Run Any .cu File in Colab

**Step 1: Enable GPU**
- Runtime → Change runtime type → T4 GPU → Save

**Step 2: Create a notebook and add this setup cell:**

```python
# Cell 1: Verify CUDA is available
!nvcc --version
!nvidia-smi
```

**Step 3: Write or paste your CUDA code:**

```python
# Cell 2: Write CUDA code to file
%%writefile device_query.cu

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               prop.totalGlobalMem / 1e9);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    }
    
    return 0;
}
```

**Step 4: Compile and run:**

```python
# Cell 3: Compile
!nvcc device_query.cu -o device_query
```

```python
# Cell 4: Run
!./device_query
```

## 📋 Standard Colab Template

Here's a template you can use for any CUDA exercise:

```python
# ============= SETUP CELL (Run First) =============
!nvcc --version
!nvidia-smi --query-gpu=name,memory.total --format=csv

# ============= CUDA CODE CELL =============
%%writefile my_program.cu

#include <stdio.h>
#include <cuda_runtime.h>

// Your CUDA code here

// ============= COMPILE CELL =============
!nvcc my_program.cu -o my_program

# ============= RUN CELL =============
!./my_program

# ============= CLEANUP (Optional) =============
!rm my_program my_program.cu
```

## 🎯 Colab GPU Specifications (Free Tier)

When you run the exercises on Colab, you'll typically get:

- **GPU**: NVIDIA Tesla T4
- **Compute Capability**: 7.5
- **CUDA Cores**: 2560
- **Memory**: 16 GB GDDR6
- **Memory Bandwidth**: 300 GB/s
- **FP32 Performance**: ~8.1 TFLOPS

## ⏱️ Colab Limitations & Tips

### Free Tier Limits
- **Session timeout**: ~90 minutes of inactivity
- **Max runtime**: 12 hours continuous
- **GPU availability**: ~12-15 hours per day (then CPU fallback)
- **No background execution**: Must keep browser tab open

### Best Practices

1. **Save frequently**: File → Save a copy in Drive
2. **Work in bursts**: Complete exercises in focused sessions
3. **Avoid long runs**: Don't run GPU for hours continuously
4. **Check GPU availability**: Run `!nvidia-smi` first
5. **Close unused sessions**: Runtime → Manage sessions

### Keeping Session Alive

Add this cell and run it to prevent timeout:

```python
# Anti-timeout cell (prevents 90-min idle disconnect)
import time
from IPython.display import clear_output

for i in range(300):  # 300 iterations = ~2.5 hours
    time.sleep(30)
    clear_output()
    print(f"Keeping session alive... {i}/300")
```

## 🔧 Common Issues & Solutions

### "No CUDA-capable device detected"

**Solution**: Enable GPU runtime
- Runtime → Change runtime type → T4 GPU → Save
- Then reconnect: Runtime → Restart runtime

### "nvcc: command not found"

**Solution**: Colab has nvcc pre-installed. Try:
```python
!/usr/local/cuda/bin/nvcc --version
```

### "Out of memory" errors

**Solution**: 
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Or restart runtime
# Runtime → Restart runtime
```

### Session disconnected

**Solution**:
- Runtime → Reconnect
- Re-run setup cells
- Your code cells are saved, just re-execute them

## 📚 Advanced: Clone Entire Repo

To work with multiple files or use Makefiles:

```python
# Clone the repo
!git clone https://github.com/sdodlapa/cuda-lab.git
%cd cuda-lab/practice/01-foundations/ex01-device-query

# Use make if Makefile exists
!make

# Or compile directly
!nvcc device_query.cu -o device_query
!./device_query
```

## 🎓 Exercise Workflow

1. **Open Colab notebook** from links above
2. **Enable GPU** (Runtime menu)
3. **Run setup cell** to verify CUDA
4. **Read the exercise** instructions
5. **Modify code** in the notebook
6. **Compile & run** to test
7. **Compare with solution** when done

## 🔗 Resources

- [Colab GPU FAQ](https://research.google.com/colaboratory/faq.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Week 1 Learning Path](../learning-path/week-01/)

## ⚡ Pro Tips

### Multiple files in one notebook

```python
# Write multiple files
%%writefile utils.h
// Header file content

%%writefile main.cu
#include "utils.h"
// Main file content

# Compile with multiple files
!nvcc main.cu -o program
```

### Compile with optimization flags

```python
!nvcc -O3 -arch=sm_75 device_query.cu -o device_query
```

### Profile your code

```python
# Use nvprof (for older CUDA)
!nvprof ./my_program

# Use ncu (for newer CUDA)
!ncu --set full ./my_program
```

---

**Ready to start?** Pick an exercise from the table at the top and click the Colab badge!
