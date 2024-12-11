/**
 * @brief This file containes kernels to generate random vectors on GPU using curand library.
 * It also containes a host function that wraps the device code.
 */

#pragma once
#include <curand_kernel.h>
#include <cuda_runtime.h>

using std::numeric_limits;

// Kernel to setup random states
__global__ void setupCurand_k(curandState *state, unsigned long seed, size_t size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// Template kernel to generate random numbers within the specified range
template <typename T>
__global__ void generateRandomVector_k(curandState *state, T *vector, size_t size, T max_magnitude) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curandState localState = state[id];
        float randomValue = curand_uniform(&localState);  // Generates a random float in [0, 1)
        bool sign = curand_uniform(&localState) > 0.5;  // Generates a random boolean
        vector[id] = (sign ? -1 : 1) * max_magnitude * randomValue;  // Scale to [-max_magnitude, max_magnitude)
        state[id] = localState;  // Save the state back
    }
}

// Host function to call the device code
template <typename T>
void generate_random_vector_gpu(T *v_gpu, size_t size_v, T max_magnitude = numeric_limits<T>::max()) {
    // Set dimensions for threads and blocks
    const size_t threadsPerBlock = 256;
    const size_t numBlocks = (size_v + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate space for random states
    curandState *devStates;
    cudaMalloc((void **)&devStates, size_v * sizeof(curandState));

    // Initialize random states with different seeds
    unsigned long long seed = time(NULL);
    setupCurand_k<<<numBlocks, threadsPerBlock>>>(devStates, seed, size_v);

    // Fill the vectors with random numbers within [min_val, max_val)
    generateRandomVector_k<<<numBlocks, threadsPerBlock>>>(devStates, v_gpu, size_v, max_magnitude);

    // Free the allocated memory for random states
    cudaFree(devStates);

    // Synchronize the device
    cudaDeviceSynchronize();
}