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
