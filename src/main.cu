#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

// Kernel definition (your implementation)
__global__ void mergeSmall_k(int* A, int* B, int* M, int sizeA, int sizeB)
{
    __shared__ int sharedA[512];
    __shared__ int sharedB[512];

    // Load A and B into shared memory
    int tid = threadIdx.x;
    if (tid < sizeA) sharedA[tid] = A[tid];
    if (tid < sizeB) sharedB[tid] = B[tid];
    __syncthreads();

    // Each thread handles one diagonal of the merge path
    int k = threadIdx.x;
    if (k < sizeA + sizeB)
    {
        int start = max(0, k - sizeB);
        int end = min(k, sizeA);
        
        // Binary search to find the intersection point
        while (start <= end)
        {
            int i = (start + end) / 2;
            int j = k - i;
            
            if (i < sizeA && j > 0 && sharedB[j-1] > sharedA[i])
                start = i + 1;
            else if (j < sizeB && i > 0 && sharedA[i-1] > sharedB[j])
                end = i - 1;
            else
            {
                // Found the intersection, write to output
                if (j >= sizeB || (i < sizeA && sharedA[i] <= sharedB[j]))
                    M[k] = sharedA[i];
                else
                    M[k] = sharedB[j];
                break;
            }
        }
    }
}

// Helper function to check for CUDA errors
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int sizeA = 500;
    const int sizeB = 400;
    const int sizeM = sizeA + sizeB;

    // Host arrays
    int *h_A = new int[sizeA];
    int *h_B = new int[sizeB];
    int *h_M = new int[sizeM];
    int *h_M_CPU = new int[sizeM];  // For CPU verification

    // Initialize A and B with sorted random numbers
    for (int i = 0; i < sizeA; i++) {
        h_A[i] = i * 2;  // Even numbers
    }
    for (int i = 0; i < sizeB; i++) {
        h_B[i] = i * 2 + 1;  // Odd numbers
    }

    // Device arrays
    int *d_A, *d_B, *d_M;
    cudaMalloc(&d_A, sizeA * sizeof(int));
    cudaMalloc(&d_B, sizeB * sizeof(int));
    cudaMalloc(&d_M, sizeM * sizeof(int));
    cudaCheckError();

    // Copy input data to device
    cudaMemcpy(d_A, h_A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Launch kernel
    dim3 block(1024);
    dim3 grid(1);
    mergeSmall_k<<<grid, block>>>(d_A, d_B, d_M, sizeA, sizeB);
    cudaCheckError();

    // Copy result back to host
    cudaMemcpy(h_M, d_M, sizeM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // CPU verification
    std::merge(h_A, h_A + sizeA, h_B, h_B + sizeB, h_M_CPU);

    // Compare results
    bool correct = true;
    for (int i = 0; i < sizeM; i++) {
        if (h_M[i] != h_M_CPU[i]) {
            correct = false;
            std::cout << "Mismatch at position " << i << ": GPU = " << h_M[i] 
                      << ", CPU = " << h_M_CPU[i] << std::endl;
            break;
        }
    }

    if (correct) {
        std::cout << "Merge successful! GPU and CPU results match." << std::endl;
    } else {
        std::cout << "Merge failed! GPU and CPU results do not match." << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_M;
    delete[] h_M_CPU;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);

    return 0;
}