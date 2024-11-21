#pragma once
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cuda_timing.h>

template <typename T>
float bench_thrust_merge(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& out, uint32_t NITER)
{
    // Create device vectors from the input std::vectors
    if(A.size()+B.size()!= out.size()) abort();
    thrust::device_vector<T> d_A = A;
    thrust::device_vector<T> d_B = B;
    thrust::device_vector<T> result(A.size() + B.size());

    float milliseconds = 0;
    TIME_EVENT_DEFINE(timing);TIME_EVENT_CREATE(timing);
    TIME_START(timing);
    for(int i = 0 ; i< NITER; i++)
    thrust::merge(thrust::device,
                  d_A.begin(), d_A.end(),
                  d_B.begin(), d_B.end(),
                  result.begin());
    TIME_STOP_SAVE(timing, milliseconds);

    TIME_EVENT_DESTROY(timing);

    thrust::copy(result.data(),result.data()+A.size() + B.size(),out.begin());
    return milliseconds/NITER;
}
