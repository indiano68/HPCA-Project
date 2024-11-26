#include <curand_kernel.h>
#include <cuda_runtime.h>

// Kernel to setup random states
__global__ void setup_kernel(curandState *state, unsigned long seed, size_t size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// Template kernel to generate random numbers within the specified range
template <typename T>
__global__ void generate_random_vector(curandState *state, T *vector, size_t size, T min, T max) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curandState localState = state[id];
        float randomValue = curand_uniform(&localState);  // Generates a random float in [0, 1)
        vector[id] = static_cast<T>(min + (max - min) * randomValue);  // Scale to [min, max)
        state[id] = localState;  // Save the state back
    }
}


// Function to generate random vectors on GPU
template <typename T>
void generate_random_vectors_cuda(T *v_A_gpu, size_t size_A, T *v_B_gpu, size_t size_B, T min_val, T max_val) {
    // Set dimensions for threads and blocks
    const size_t threadsPerBlock = 256;
    const size_t blocks_A = (size_A + threadsPerBlock - 1) / threadsPerBlock;
    const size_t blocks_B = (size_B + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate space for random states
    curandState *devStates_A, *devStates_B;
    cudaMalloc((void **)&devStates_A, size_A * sizeof(curandState));
    cudaMalloc((void **)&devStates_B, size_B * sizeof(curandState));

    // Initialize random states with different seeds
    unsigned long long seed = time(NULL);
    setup_kernel<<<blocks_A, threadsPerBlock>>>(devStates_A, seed, size_A);
    setup_kernel<<<blocks_B, threadsPerBlock>>>(devStates_B, seed + 1, size_B);

    // Fill the vectors with random numbers within [min_val, max_val)
    generate_random_vector<<<blocks_A, threadsPerBlock>>>(devStates_A, v_A_gpu, size_A, min_val, max_val);
    generate_random_vector<<<blocks_B, threadsPerBlock>>>(devStates_B, v_B_gpu, size_B, min_val, max_val);

    cudaDeviceSynchronize();

    // Free the allocated memory for random states
    cudaFree(devStates_A);
    cudaFree(devStates_B);
}