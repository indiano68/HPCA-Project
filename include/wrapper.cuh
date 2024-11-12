#pragma once
#include <path_merge.cuh>

__global__ void empty_kernel()
{
    return;
}

template<typename T>
const std::vector<T> call_merge_kernel(std::vector<T> A,std::vector<T> B)
{

    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");

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
    empty_kernel<<<1,1>>>();

    T * A_dev, * B_dev, * M_dev;

    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "B.size() = " << B.size() << std::endl;

    //time

    cudaEvent_t start, stop;
    cudaEvent_t malloc_start, malloc_stop;
    cudaEvent_t memcpy_HtD_start, memcpy_HtD_stop;
    cudaEvent_t kernel_start, kernel_stop;
    cudaEvent_t memcpy_DtH_start, memcpy_DtH_stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&malloc_start);
    cudaEventCreate(&malloc_stop);
    cudaEventCreate(&memcpy_HtD_start);
    cudaEventCreate(&memcpy_HtD_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&memcpy_DtH_start);
    cudaEventCreate(&memcpy_DtH_stop);

    // Time CUDA malloc operations
    cudaEventRecord(malloc_start);
    cudaMalloc(&A_dev, A.size() * sizeof(T));
    cudaMalloc(&B_dev, B.size() * sizeof(T));
    cudaMalloc(&M_dev, (A.size() + B.size()) * sizeof(T));
    cudaEventRecord(malloc_stop), cudaEventSynchronize(malloc_stop);

    // Time CUDA memcpy Host to Device
    cudaEventRecord(memcpy_HtD_start);
    cudaMemcpy(A_dev, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(memcpy_HtD_stop), cudaEventSynchronize(memcpy_HtD_stop);
  
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

    cudaEventRecord(kernel_start);
    merge_k_gpu_triangles<<<grid, block>>>(A_dev, A.size(),
                                                 B_dev, B.size(),
                                                 M_dev);

    cudaEventRecord(kernel_stop), cudaEventSynchronize(kernel_stop);

    cudaEventRecord(memcpy_DtH_start);
    cudaMemcpy(M.data(), M_dev, M.size() * sizeof(T), cudaMemcpyDeviceToHost);
    cudaEventRecord(memcpy_DtH_stop), cudaEventSynchronize(memcpy_DtH_stop);

    // Record overall stop time
    cudaEventRecord(stop), cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, malloc_start, malloc_stop);
    std::cout << "CUDA malloc time: " << milliseconds << "ms" << std::endl;

    cudaEventElapsedTime(&milliseconds, memcpy_HtD_start, memcpy_HtD_stop);
    std::cout << "CUDA memcpy HtD time: " << milliseconds << "ms" << std::endl;

    cudaEventElapsedTime(&milliseconds, kernel_start, kernel_stop);
    std::cout << "CUDA kernel time: " << milliseconds << "ms" << std::endl;

    cudaEventElapsedTime(&milliseconds, memcpy_DtH_start, memcpy_DtH_stop);
    std::cout << "CUDA memcpy DtH time: " << milliseconds << "ms" << std::endl;

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total CUDA operation time: " << milliseconds << "ms" << std::endl;

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(M_dev);

    return M;
  
}



