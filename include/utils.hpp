// Description: Utility functions for the project.

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <stdio.h> // printf

using std::numeric_limits;

// CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void emptyk(){}; // Empty kernel to warm up the GPU before timing our kernels

template<typename T>
inline size_t vector_sizeof(const std::vector<T>& vector)
{
    return sizeof(T)*vector.size();
}

template<typename T>
void generate_random_vector_cpu(std::vector<T>& vector, size_t size, T min = numeric_limits<T>::min(), T max = numeric_limits<T>::max())
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min, max);
  std::generate(vector.begin(), vector.end(), [&]() { return static_cast<T>(dis(gen)); });
}

// Print GPU information
void printGPUInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
               prop.memoryClockRate / 1024);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");
    }
}