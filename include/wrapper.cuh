#pragma once
#include <path_merge.cuh>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/execution_policy.h>

_global_ void empty_k()
{
    return;
}

template<typename T>
std::vector<T> call_merge_kernel(std::vector<T> A,std::vector<T> B)
{

    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");

    std::cout << "\n----------- call_merge_kernel wrapper -----------" << std::endl;

    std::vector<T> M(A.size() + B.size());

    if(M.size() < THREADS_PER_BLOCK)
    {
      std::cout << "M is small enough to be merged on the CPU (size = " << M.size() << ")" << std::endl;
      return mergeSmall_k_cpu(A, B);
    } 

    if(A.size() > B.size())
    {
      std::swap(A,B);
    }

    int num_blocks = (M.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto remainder = M.size() % THREADS_PER_BLOCK;
    size_t padding = (remainder == 0) ? 0 : THREADS_PER_BLOCK - remainder;

    if(remainder != 0)
    {
      std::cout << "THREADS_PER_BLOCK does not evenly divide M.size(). "
                << "Adding " << padding << " elements to B." << std::endl;
      T biggest_element = std::max(A.back(), B.back());
      //fill the rest of B with the biggest element
      B.resize(B.size() + padding, biggest_element);
    }

    //call empty kernel to initialize the GPU
    // empty_k<<<1,1>>>();

    T * A_dev, * B_dev, * M_dev;
    int2 * Q_global;

    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "B.size() = " << B.size() << std::endl;

    dim3 block(THREADS_PER_BLOCK), grid(num_blocks);

    std::cout << "Launching kernel with " << num_blocks << " blocks and " << THREADS_PER_BLOCK << " threads per block" << std::endl;

    cudaEvent_t start, before_memcpyHtD, before_kernel, before_memcpyDtH, stop;
    //cudaEvent_t before_kernel2;

    cudaEventCreate(&start);
    cudaEventCreate(&before_memcpyHtD);
    cudaEventCreate(&before_kernel);
    //cudaEventCreate(&before_kernel2);
    cudaEventCreate(&before_memcpyDtH);
    cudaEventCreate(&stop);

    // Time CUDA malloc operations
    cudaEventRecord(start);
    cudaMalloc(&A_dev, A.size() * sizeof(T));
    cudaMalloc(&B_dev, B.size() * sizeof(T));
    cudaMalloc(&M_dev, (A.size() + B.size()) * sizeof(T));
    cudaMalloc(&Q_global, grid.x * sizeof(int2));

    cudaEventRecord(before_memcpyHtD), cudaEventSynchronize(before_memcpyHtD);
    cudaMemcpy(A_dev, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice);

    cudaEventRecord(before_kernel), cudaEventSynchronize(before_kernel);
    // merge_k_gpu_triangles<<<grid, block>>>(A_dev, A.size(),
    //                                        B_dev, B.size(),
    //                                        M_dev);

    merge_k_gpu_squares_v2<<<grid, block>>>(A_dev, A.size(),
                                           B_dev, B.size(),
                                           M_dev);

    // partition_k_gpu<<<grid, 1>>>(A_dev, A.size(),
    //                             B_dev, B.size(),
    //                             Q_global);
    
    // cudaEventRecord(before_kernel2), cudaEventSynchronize(before_kernel2);

    // merge_k_gpu_squares<<<grid, block>>>(A_dev, A.size(),
    //                                      B_dev, B.size(),
    //                                      M_dev, Q_global);

    cudaEventRecord(before_memcpyDtH), cudaEventSynchronize(before_memcpyDtH);
    cudaMemcpy(M.data(), M_dev, M.size() * sizeof(T), cudaMemcpyDeviceToHost);

    // Record overall stop time
    cudaEventRecord(stop), cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, before_memcpyHtD);
    std::cout << "CUDA malloc time: " << milliseconds << "ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, before_memcpyHtD, before_kernel);
    std::cout << "CUDA memcpy HtD time: " << milliseconds << "ms" << std::endl;
    // cudaEventElapsedTime(&milliseconds, before_kernel, before_kernel2);
    // std::cout << "Partition time: " << milliseconds << "ms" << std::endl;
    // cudaEventElapsedTime(&milliseconds, before_kernel2, before_memcpyDtH);
    // std::cout << "Merge time: " << milliseconds << "ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, before_kernel, before_memcpyDtH);
    std::cout << "CUDA kernel time: " << milliseconds << "ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, before_memcpyDtH, stop);
    std::cout << "CUDA memcpy DtH time: " << milliseconds << "ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA Total merge time: " << milliseconds << "ms" << std::endl;

    std::cout << "----------- end call_merge_kernel wrapper -----------" << std::endl;

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(M_dev);

    return M;
  
}

template <typename T>
std::vector<T> merge_arrays_thrust(const std::vector<T>& A, const std::vector<T>& B)
{

    std::cout << "\n----------- merge_array_thrust wrapper -----------" << std::endl;

    cudaEvent_t start, before_kernel, before_memcpyDtH, stop;
    cudaEventCreate(&start), cudaEventCreate(&before_kernel), cudaEventCreate(&before_memcpyDtH), cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Create device vectors from the input std::vectors
    thrust::device_vector<T> d_A = A;
    thrust::device_vector<T> d_B = B;

    thrust::device_vector<T> result(A.size() + B.size());

    thrust::copy(A.begin(), A.end(), d_A.begin());
    thrust::copy(B.begin(), B.end(), d_B.begin());


    cudaEventRecord(before_kernel), cudaEventSynchronize(before_kernel);
    thrust::merge(thrust::device,
                  d_A.begin(), d_A.end(),
                  d_B.begin(), d_B.end(),
                  result.begin());

    cudaEventRecord(before_memcpyDtH), cudaEventSynchronize(before_memcpyDtH);

    auto result_host = std::vector<T>(result.size());
    thrust::copy(result.begin(), result.end(), result_host.begin());

    cudaEventRecord(stop), cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, before_kernel);
    std::cout << "CUDA malloc + memcpy HtD time: " << milliseconds << " ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, before_kernel, before_memcpyDtH);
    std::cout << "Kernel time: " << milliseconds << " ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, before_memcpyDtH, stop);
    std::cout << "CUDA memcpy DtH time: " << milliseconds << " ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA Total merge time: " << milliseconds << " ms" << std::endl;

    std::cout << "----------- end merge_array_thrust wrapper -----------\n" << std::endl;

    return result_host;
}