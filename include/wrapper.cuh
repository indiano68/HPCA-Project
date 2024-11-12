#pragma once
#include <path_merge.cuh>

template<typename T>
const std::vector<T> call_merge_kernel(std::vector<T> A,std::vector<T> B)
{

    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");

    // std::sort(A.begin(),A.end());
    // std::sort(B.begin(),B.end());

    std::vector<T> M(A.size() + B.size());

    if(M.size() < THREADS_PER_BLOCK)
    {
      std::cout << "M is small enough to be merged on the CPU (size = " << M.size() << ")" << std::endl;
      return mergeSmall_k_cpu(A, B);
    } 

    //std::vector<double> A = build_random_vector<double>(764, 1, 10);
    //std::vector<double> B = build_random_vector<double>(934, 1, 10);

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

    T * A_dev, * B_dev, * M_dev;

    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "B.size() = " << B.size() << std::endl;

    cudaMalloc(&A_dev, A.size() * sizeof(T));
    cudaMalloc(&B_dev, B.size() * sizeof(T));
    cudaMalloc(&M_dev, (A.size() + B.size()) * sizeof(T));

    cudaMemcpy(A_dev, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice);

    if(DEBUG)
    {
      std::cout << "A_dev = [";
      for(int i = 0; i < A.size(); i++)
      {
        std::cout << A[i] << ", ";
      }
      std::cout << "]" << std::endl;

      std::cout << "B_dev = [";
      for(int i = 0; i < B.size(); i++)
      {
        std::cout << B[i] << ", ";
      }
      std::cout << "]" << std::endl;
    }

    dim3 block(THREADS_PER_BLOCK), grid(num_blocks);
    //size_t shared_mem_size = (2 * THREADS_PER_BLOCK + 2) * sizeof(T) + sizeof(int2);

    std::cout << "Launching kernel with " << num_blocks << " blocks and " << THREADS_PER_BLOCK << " threads per block" << std::endl;

    //time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mergeSmall_k_gpu_multiblock<<<grid, block>>>(A_dev, A.size(),
                                                 B_dev, B.size(),
                                                 M_dev);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU merge time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(M.data(), M_dev, M.size() * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(M_dev);

    return M;
  
}



