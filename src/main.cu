#include<iostream>
#include<stdio.h>
#include<tools.hpp>
#include<vector>
#include<algorithm>
#include<chrono>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/execution_policy.h>
#include<path_merge.cuh>

#include <wrapper.cuh>

template <typename T>
std::vector<T> merge_arrays_thrust(const std::vector<T>& A, const std::vector<T>& B)
{
    // Create device vectors from the input std::vectors
    thrust::device_vector<T> d_A = A;
    thrust::device_vector<T> d_B = B;

    thrust::device_vector<T> result(A.size() + B.size());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::merge(thrust::device,
                  d_A.begin(), d_A.end(),
                  d_B.begin(), d_B.end(),
                  result.begin());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA Thrust merge time: " << milliseconds << "ms" << std::endl;

    // Convert the result back to std::vector and return
    return std::vector<T>(result.begin(), result.end());
}

int main()
{

    printGPUInfo();

    std::vector<double> A = build_random_vector<double>(1000000, 1, 10);
    std::vector<double> B = build_random_vector<double>(100000, 1, 10);

    std::sort(A.begin(),A.end());
    std::sort(B.begin(),B.end());

    auto gpu_merge = call_merge_kernel(A, B);

    auto thrust_merge = merge_arrays_thrust(A, B);

    /*
    auto start = std::chrono::high_resolution_clock::now();

    auto cpu_merge  = mergeSmall_k_cpu(A, B);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU merge time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    */

    //compare the results
    if(std::equal(gpu_merge.begin(), gpu_merge.end(), thrust_merge.begin()))
    {
        std::cout<<"TEST PASSED!"<<std::endl;
    }
    else
    {
        std::cout<<"GPU and CPU results are not equal"<<std::endl;
    }

    // print_vector(gpu_merge, "GPU merge result");
    // print_vector(cpu_merge, "CPU merge result");

    return;

}
