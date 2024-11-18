#pragma once
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/execution_policy.h>

template <typename T>
float bench_thrust_merge(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& out, uint32_t NITER)
{
    // Create device vectors from the input std::vectors
    if(A.size()+B.size()!= out.size()) abort();
    float milliseconds = 0;
    thrust::device_vector<T> d_A = A;
    thrust::device_vector<T> d_B = B;
    thrust::device_vector<T> result(A.size() + B.size());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0 ; i< NITER; i++)
    thrust::merge(thrust::device,
                  d_A.begin(), d_A.end(),
                  d_B.begin(), d_B.end(),
                  result.begin());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::vector<T>result_out(A.size() + B.size());
    thrust::copy(result.data(),result.data()+A.size() + B.size(),out.begin());
    return milliseconds/NITER;
}
