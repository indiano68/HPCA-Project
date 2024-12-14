// Purpose: Main file for benchmarking the sortSmallBatch_k kernel

#include <iostream>
#include <string>
#include <batch_sort.cuh>
#include <utils.hpp>
#include <cassert>
#include <cuda_timing.h>

// data type used in the sorting
using v_type = int;

int main(int argc, char **argv)
{
  printGPUInfo();
  if (argc < 3)
  {
      std::cout << "Usage: " << argv[0] << " <N> " << " <d> " << std::endl;
      abort();
  }

  // Parse command line arguments and ensure they are valid
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

  // Generate random batches to sort
  std::vector<v_type> batches(N * d);
  generate_random_vector_cpu(batches, N * d);

  std::vector<v_type> vector_out(N * d);
  v_type *v_M_gpu;

  //timing events
  TIME_EVENT_DEFINE(timing_kernel);TIME_EVENT_CREATE(timing_kernel);
  float timing_kernel_ms = 0.0f;

  cudaMalloc(&v_M_gpu, vector_sizeof(batches));
  cudaMemcpy(v_M_gpu, batches.data(), vector_sizeof(batches), cudaMemcpyHostToDevice);

  //define number of threads and blocks
  unsigned num_threads = min(MAX_BATCH_SIZE, N * d);
  unsigned num_blocks = (N * d + num_threads - 1) / num_threads;
  std::cout << "Sorting " << N << " batches of size " << d << std::endl;
  std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;

  //kernel call
  emptyk<<<1, 1>>>();
  TIME_START(timing_kernel);
  sortSmallBatch_k<<<num_blocks, num_threads>>>(v_M_gpu, N, d);
  TIME_STOP_SAVE(timing_kernel, timing_kernel_ms);
  //copy back the result to the host
  cudaMemcpy(vector_out.data(), v_M_gpu, vector_sizeof(batches), cudaMemcpyDeviceToHost);

  //execute the CPU version and compare the results
  float cpu_time = sort_batch_cpu(batches, N, d);
  std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
  std::cout << "Kernel time: " << timing_kernel_ms << " ms" << std::endl;
  std::cout << "Equality: " << (batches == vector_out ? "TRUE" : "FALSE") << std::endl;

  cudaFree(v_M_gpu);
  TIME_EVENT_DESTROY(timing_kernel);

  return EXIT_SUCCESS;
}