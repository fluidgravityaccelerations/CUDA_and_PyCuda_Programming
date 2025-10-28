# CUDA and PyCuda Programming

## CUDA Array Sum Example (Simple_CUDA_Google_Colab.ipynb)

The file contains a simple example of a **CUDA program** that performs element-wise array summation on the GPU. The example is implemented as a **Google Colab Jupyter notebook**, demonstrating how to compile and run CUDA code directly within the Colab environment. It serves as a minimal reference for learning how to allocate memory on the GPU, launch kernels, and manage CUDA error checking.

## PyCUDA Vector Addition Example Version #2 (fiveDifferentWaysToSumVectorsInPyCUDA_version1.ipynb)

Example demonstrating how to execute a **CUDA kernel** from within a **Google Colab** environment using **PyCUDA**. The code performs the **elementwise summation of two vectors** on the GPU.

The implementation is based on the compilation and execution of a custom CUDA kernel through `SourceModule`

## PyCUDA Vector Addition Example Version #2 (fiveDifferentWaysToSumVectorsInPyCUDA_version2.ipynb)

Example demonstrating how to perform **vector addition on the GPU** using **PyCUDA’s `SourceModule`**, while transferring data **on-the-fly** between host (CPU) and device (GPU).

Unlike Version #1, which explicitly allocates GPU memory and manually copies arrays, this version leverages the convenience of `cuda.In()` and `cuda.Out()`:
- `cuda.In()` automatically copies NumPy arrays from host to device before kernel execution.  
- `cuda.Out()` copies the computed result from device to host after execution.  

This approach simplifies memory management at the cost of a small overhead due to temporary allocations.  

## PyCUDA Vector Addition Example Version #3 (fiveDifferentWaysToSumVectorsInPyCUDA_version3.ipynb)

Example demonstrating a more **high-level PyCUDA approach** to perform vector addition on the GPU using the `gpuarray` class.  
Unlike the previous implementations, which explicitly handled memory allocation and data transfer, this version relies on **automatic GPU memory management** provided by PyCUDA.

The workflow includes:
- Transferring NumPy arrays to the GPU using `gpuarray.to_gpu()`
- Performing the vector addition directly with `d_c = d_a + d_b`
- Retrieving the result back to host memory with `.get()`

No explicit kernel definition or memory allocation is required — PyCUDA’s `gpuarray` handles these operations internally, providing a clean and concise syntax for GPU-based computations.
