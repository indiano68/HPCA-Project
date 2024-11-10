#include<iostream>
#include<stdio.h>
#include<tools.hpp>
#include<vector>
#include<algorithm>
#include<path_merge.cuh>

#define THREADS_PER_BLOCK 5

int main()
{

    printGPUInfo();

    //std::vector<double> A    = build_random_vector<double>(4,0,100);
    //std::vector<double> B    = build_random_vector<double>(6,0,100);
    // std::vector<double> A = {1, 3, 5, 7};
    // std::vector<double> B = {0, 2, 4, 6, 8, 10};

    std::vector<double> A = {1, 3, 5, 7, 8, 50, 4, 21, 41, 52};
    std::vector<double> B = {0, 2, 4, 6, 8, 10, 11, 49, 51, 53};

    std::vector<double> M(A.size()+B.size());
    double * A_dev, * B_dev, * M_dev;

    std::sort(A.begin(),A.end());
    std::sort(B.begin(),B.end());

    cudaMalloc(&A_dev,A.size()    *sizeof(double));
    cudaMalloc(&B_dev,B.size()    *sizeof(double));
    cudaMalloc(&M_dev,M.size()*sizeof(double));

    cudaMemcpy(A_dev,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev,B.data(),B.size()*sizeof(double),cudaMemcpyHostToDevice);

    // dim3 block(M.size()), grid(1);

    // mergeSmall_k_gpu_v2<<<grid, block>>>(A_dev,A.size(),
    //                           B_dev,B.size(),
    //                           M_dev,M.size());

    constexpr int block_size = THREADS_PER_BLOCK;
    dim3 grid(4), block(block_size);
    mergeSmall_k_gpu_multiblock<<<grid, block>>>(A_dev,A.size(),
                              B_dev,B.size(),
                              M_dev,M.size());

    //check for kernel errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(M.data(),M_dev,M.size()*sizeof(double),cudaMemcpyDeviceToHost);                         

    print_vector(A, "A = ");
    print_vector(B, "B = ");
    auto cpu_merge  = mergeSmall_k_cpu(A, B);
    // print_vector(cpu_merge, "CPU merge: ");
    print_vector(M, "GPU merge: ");

}
