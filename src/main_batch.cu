#include <iostream>
#include <tools.hpp>
#include <batch_merge.cuh>
#include <batch_merge_tools.hpp>
#define CUDA_TIMING
#include <cuda_timing.h>

using v_type = int;

__global__ void emptyk(){};
static constexpr unsigned short MAX_BATCH_SIZE = 1024;

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
  unsigned short d = static_cast<unsigned short>(d_arg);

  std::vector<v_type> A_B_vectors(N * d);
  std::vector<unsigned short> A_sizes = build_and_sort_batches(A_B_vectors, N, d, -10, 10);

  std::vector<v_type> vector_out(N * d);
  v_type *v_A_B_gpu;
  unsigned short *v_A_sizes_gpu;

  TIME_EVENT_DEFINE(timing_kernel);TIME_EVENT_CREATE(timing_kernel);
  float timing_kernel_ms = 0.0f;

  cudaMalloc(&v_A_B_gpu, vector_sizeof(A_B_vectors));
  cudaMalloc(&v_A_sizes_gpu, vector_sizeof(A_sizes));
  cudaMemcpy(v_A_B_gpu, A_B_vectors.data(), vector_sizeof(A_B_vectors), cudaMemcpyHostToDevice);
  cudaMemcpy(v_A_sizes_gpu, A_sizes.data(), vector_sizeof(A_sizes), cudaMemcpyHostToDevice);

  unsigned num_threads = min((MAX_BATCH_SIZE / d) * d, N * d);
  unsigned num_blocks = (N * d + num_threads - 1) / num_threads;
  std::cout << "num_threads: " << num_threads << ", num_blocks: " << num_blocks << std::endl;

  //kernel call
  emptyk<<<1, 1>>>();
  TIME_START(timing_kernel);
  mergeSmallBatch_k<<<num_blocks, num_threads>>>(v_A_B_gpu, v_A_sizes_gpu, N, d);
  TIME_STOP_SAVE(timing_kernel, timing_kernel_ms);

  cudaMemcpy(vector_out.data(), v_A_B_gpu, vector_sizeof(A_B_vectors), cudaMemcpyDeviceToHost);

  merge_batch_cpu(A_B_vectors, A_sizes, N, d);
  std::cout << "Kernel time: " << timing_kernel_ms << " ms" << std::endl;
  std::cout << "Equality: " << (A_B_vectors == vector_out ? "TRUE" : "FALSE") << std::endl;

  cudaFree(v_A_B_gpu);
  cudaFree(v_A_sizes_gpu);
  TIME_EVENT_DESTROY(timing_kernel);

  return;
}