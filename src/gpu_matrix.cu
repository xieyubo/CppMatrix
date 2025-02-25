#include "std_patch.h"
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

__global__ void vectorPow(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = powf(A[i], B[i]);
    }
}

void CudaFree(void* p)
{
    if (p) {
        auto err = cudaFree(p);
        if (err != cudaSuccess) {
            throw std::runtime_error { "cudaFree failed." };
        }
    }
}

template <typename T>
std::unique_ptr<T, decltype(&CudaFree)> CudaMalloc(size_t numElements)
{
    T* p {};
    auto err = cudaMalloc((void**)&p, sizeof(T) * numElements);
    if (err != cudaSuccess) {
        throw std::runtime_error { "cudaMalloc failed." };
    }
    return { p, &CudaFree };
}

void CudaPow(std::float16_t* inputA, std::float16_t* inputB, std::float16_t* output, size_t bufferSize)
{
    throw std::runtime_error { "Not implemented." };
}

void CudaPow(std::float32_t* inputA, std::float32_t* inputB, std::float32_t* output, size_t bufferSize)
{
    auto numElements = bufferSize / sizeof(std::float32_t);

    // Allocate the device input vector A.
    auto d_A = CudaMalloc<std::float32_t>(numElements);

    // Allocate the device input vector B.
    auto d_B = CudaMalloc<std::float32_t>(numElements);

    // Allocate the output vector.
    auto d_C = CudaMalloc<std::float32_t>(numElements);

    // Copy host to device.
    auto err = cudaMemcpy(d_A.get(), inputA, bufferSize, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B.get(), inputB, bufferSize, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorPow<<<blocksPerGrid, threadsPerBlock>>>(d_A.get(), d_B.get(), d_C.get(), numElements);
    err = cudaGetLastError();

    // Copy from device to host.
    err = cudaMemcpy(output, d_C.get(), bufferSize, cudaMemcpyDeviceToHost);
}