#include <iostream>
#include <utils.hpp>
#include <batch_sort.cuh>
#include <string>

#define CUDA_TIMING
#include <cuda_timing.h>

using v_type = int;

__global__ void emptyk(){};

#define MAX_BATCH_SIZE 1024

#define for_merge false

int main(int argc, char **argv)
{

  if (argc < 3)
  {
      std::cout << "Usage: " << argv[0] << " <N> " << " <d> " << std::endl;
      abort();
  }

  unsigned N = std::stoi(argv[1]);
  unsigned d_arg = std::stoi(argv[2]);
  if(d_arg > MAX_BATCH_SIZE || d_arg < 2)
  {
    std::cout << "d must be between 2 and 1024" << std::endl;
    abort();
  }
  else if(d_arg & (d_arg - 1))
  {
    std::cout << "d must be a power of 2" << std::endl;
    abort();
  }
  
  unsigned short d = static_cast<unsigned short>(d_arg);

  std::vector<v_type> batches(N * d);
  generate_random_vector_cpu(batches, N * d);

  std::vector<v_type> vector_out(N * d);
  v_type *v_M_gpu;

  TIME_EVENT_DEFINE(timing_kernel);TIME_EVENT_CREATE(timing_kernel);
  float timing_kernel_ms = 0.0f;

  cudaMalloc(&v_M_gpu, vector_sizeof(batches));
  cudaMemcpy(v_M_gpu, batches.data(), vector_sizeof(batches), cudaMemcpyHostToDevice);

  unsigned num_threads = min((MAX_BATCH_SIZE / d) * d, N * d);
  unsigned num_blocks = (N * d + num_threads - 1) / num_threads;
  std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;

  //kernel call
  emptyk<<<1, 1>>>();
  TIME_START(timing_kernel);
  if constexpr(for_merge)
  {
    for(unsigned merge_lenght = 2; merge_lenght <= d; merge_lenght *= 2)
    {
      unsigned N_merge = N * d / merge_lenght;
      mergeSmallBatch_for_k<<<num_blocks, num_threads>>>(v_M_gpu, N_merge, merge_lenght); 
    }
  }
  else
  {
    sortSmallBatch_k<<<num_blocks, num_threads>>>(v_M_gpu, N, d);
  }
  TIME_STOP_SAVE(timing_kernel, timing_kernel_ms);

  cudaMemcpy(vector_out.data(), v_M_gpu, vector_sizeof(batches), cudaMemcpyDeviceToHost);

  sort_batch_cpu(batches, N, d);
  std::cout << "Kernel time: " << timing_kernel_ms << " ms" << std::endl;
  std::cout << "Equality: " << (batches == vector_out ? "TRUE" : "FALSE") << std::endl;

  cudaFree(v_M_gpu);
  TIME_EVENT_DESTROY(timing_kernel);

  return EXIT_SUCCESS;
}