# 🚀 Google Colab Quick Start Guide

Run CUDA code on **free T4 GPU** in your browser - no installation required!

## ⚡ 60-Second Setup

### Option 1: Pre-Made Notebooks (Easiest)

Click any badge to launch:

**Learning Path (Week 1)**
- Day 1: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-1-gpu-basics.ipynb) GPU Basics
- Day 2: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-2-thread-indexing.ipynb) Thread Indexing
- Day 3: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-3-memory-basics.ipynb) Memory Basics
- Day 4: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/learning-path/week-01/day-4-error-handling.ipynb) Error Handling

**Practice Exercises**
- Ex01: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex01-device-query/colab-device-query.ipynb) Device Query
- Ex02: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapa/cuda-lab/blob/main/practice/01-foundations/ex02-hello-gpu/colab-hello-gpu.ipynb) Hello GPU

### Option 2: Run Any .cu File

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Enable GPU: **Runtime → Change runtime type → T4 GPU → Save**
4. Use this template:

```python
# ========== Cell 1: Verify CUDA ==========
!nvcc --version
!nvidia-smi

# ========== Cell 2: Write CUDA code ==========
%%writefile my_program.cu

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel() {
    printf("Hello from thread %d\\n", threadIdx.x);
}

int main() {
    myKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}

# ========== Cell 3: Compile ==========
!nvcc my_program.cu -o my_program

# ========== Cell 4: Run ==========
!./my_program
```

---

## 📊 What You Get (Free Tier)

### GPU Specs
- **GPU**: NVIDIA Tesla T4
- **CUDA Cores**: 2560
- **Memory**: 16 GB GDDR6
- **Compute Capability**: 7.5
- **Memory Bandwidth**: 300 GB/s
- **FP32 Performance**: ~8.1 TFLOPS

### Usage Limits
- **Session**: ~90 min idle timeout
- **Max runtime**: 12 hours continuous
- **GPU hours**: ~12-15 hours per day (free)
- **Storage**: Temporary (files deleted after session)

---

## 💡 Essential Tips

### 1. Save Your Work
```python
# Mount Google Drive (do this first!)
from google.colab import drive
drive.mount('/content/drive')

# Save files to Drive
!cp my_program.cu /content/drive/MyDrive/cuda-programs/
```

### 2. Keep Session Alive
Add this cell to prevent timeout:
```python
import time
from IPython.display import clear_output

for i in range(180):  # 90 minutes
    time.sleep(30)
    clear_output()
    print(f"⏰ Session active: {i}/180")
```

### 3. Check GPU Availability
```python
# Verify GPU is available
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

### 4. Work with Multiple Files
```python
%%writefile utils.h
// Header file

%%writefile main.cu
#include "utils.h"
// Main code

!nvcc main.cu -o program
!./program
```

### 5. Clone This Repo
```python
!git clone https://github.com/sdodlapa/cuda-lab.git
%cd cuda-lab
!ls -la
```

---

## 🔧 Common Issues & Solutions

### ❌ "No CUDA-capable device detected"
**Solution**: Enable GPU runtime
- Runtime → Change runtime type → T4 GPU → Save
- Then: Runtime → Restart runtime

### ❌ "nvcc: command not found"
**Solution**: Use full path
```python
!/usr/local/cuda/bin/nvcc --version
```

### ❌ "Out of memory"
**Solution**: Clear GPU memory
```python
import torch
torch.cuda.empty_cache()
```
Or: Runtime → Restart runtime

### ❌ Session disconnected
**Solution**: Just reconnect
- Runtime → Reconnect
- Re-run setup cells (they're saved!)

### ❌ No GPU available (after using for hours)
**Solution**: You've hit daily limit
- Wait ~12 hours
- Or upgrade to Colab Pro ($10/month)
- Or use CPU runtime for non-GPU tasks

---

## 📚 Complete Workflow Example

Here's a complete example for a vector addition kernel:

```python
# ========== Setup ==========
!nvcc --version
print("\\nGPU Info:")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ========== CUDA Code ==========
%%writefile vector_add.cu

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            success = false;
            break;
        }
    }
    
    printf("Result: %s\\n", success ? "PASS" : "FAIL");
    printf("Added %d vectors of size %d each\\n", 2, n);
    
    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}

# ========== Compile ==========
!nvcc vector_add.cu -o vector_add

# ========== Run ==========
!./vector_add

# ========== Cleanup ==========
!rm vector_add vector_add.cu
```

---

## 🎯 Best Practices

### DO ✅
- Save work frequently (to Drive)
- Enable GPU before starting
- Use `cudaDeviceSynchronize()` before checking results
- Check CUDA errors
- Keep browser tab open to avoid timeout
- Work in focused bursts (not 12hr marathons)

### DON'T ❌
- Don't expect CPU-speed file I/O
- Don't mine cryptocurrency (will get banned)
- Don't run computations for 12 hours straight
- Don't forget to free memory
- Don't assume sequential thread execution

---

## 🚀 Advanced Features

### Timing Your Code
```python
%%writefile timed_kernel.cu

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Your kernel launch here
    myKernel<<<blocks, threads>>>();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel time: %.3f ms\\n", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
```

### Profiling with nvprof
```python
# Compile with debug info
!nvcc -lineinfo my_program.cu -o my_program

# Profile
!nvprof ./my_program

# Or use ncu (Nsight Compute)
!ncu --set full --export profile ./my_program
```

### Using cuBLAS/cuFFT/etc
```python
# Example: Matrix multiplication with cuBLAS
!nvcc -lcublas my_program.cu -o my_program
```

---

## 📱 Mobile Users

Colab works on mobile browsers, but:
- Use landscape mode
- External keyboard recommended
- May have GPU availability issues
- Consider using tablet/desktop

---

## 💰 Colab Pro (Optional)

If you need more:
- **Colab Pro** ($10/month): More GPU time, faster GPUs (P100/V100), longer sessions
- **Colab Pro+** ($50/month): Even more resources, background execution

Free tier is usually sufficient for learning!

---

## 🔗 Resources

- **[Learning Path](learning-path/README.md)** - Start learning
- **[Practice Exercises](practice/COLAB-SETUP.md)** - Detailed exercise guide
- **[CUDA Guide](cuda-programming-guide/index.md)** - Reference docs
- **[Colab FAQ](https://research.google.com/colaboratory/faq.html)** - Official docs

---

## 🎓 Getting Started Checklist

- [ ] Click a Colab badge above
- [ ] Enable T4 GPU runtime
- [ ] Run first cell to verify CUDA
- [ ] Complete your first exercise
- [ ] Save notebook to Drive
- [ ] Move to next lesson

**Ready?** Click a badge above and start coding! 🚀

---

**Pro Tip**: Work through Week 1 notebooks first, then practice with exercises. The notebooks explain concepts, exercises let you apply them!
