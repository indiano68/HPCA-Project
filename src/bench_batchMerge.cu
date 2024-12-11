#include <iostream>
#include <string>
#include <batch_merge.cuh>
#include <cuda_timing.h>
#include <utils.hpp>

using v_type = float;

int main(int argc, char **argv)
{

  if (argc < 3)
  {
    std::cout << "Usage: " << argv[0] << " <N> " << " <d> " << std::endl;
    abort();
  }

  unsigned N = std::stoi(argv[1]);
  unsigned d_arg = std::stoi(argv[2]);
  if (d_arg > MAX_BATCH_SIZE || d_arg < 2)
  {
    std::cout << "d must be between 2 and 1024" << std::endl;
    abort();
  }
  unsigned short d = static_cast<unsigned short>(d_arg);

  std::vector<v_type> A_B_vectors(N * d);
  std::vector<unsigned short> A_sizes = build_and_sort_batches(A_B_vectors, N, d);

  std::vector<v_type> vector_out(N * d);
  v_type *v_A_B_gpu;
  v_type *v_A_B_buffer;
  unsigned short *v_A_sizes_gpu;

  TIME_EVENT_DEFINE(timing_kernel);
  TIME_EVENT_CREATE(timing_kernel);
  float time_kernel_ms = 0.0f;

  cudaMalloc(&v_A_B_gpu, vector_sizeof(A_B_vectors));
  cudaMalloc(&v_A_B_buffer, vector_sizeof(A_B_vectors));

  cudaMalloc(&v_A_sizes_gpu, vector_sizeof(A_sizes));
  cudaMemcpy(v_A_B_gpu, A_B_vectors.data(), vector_sizeof(A_B_vectors), cudaMemcpyHostToDevice);
  cudaMemcpy(v_A_B_buffer, v_A_B_gpu, vector_sizeof(A_B_vectors), cudaMemcpyDeviceToDevice);

  cudaMemcpy(v_A_sizes_gpu, A_sizes.data(), vector_sizeof(A_sizes), cudaMemcpyHostToDevice);

  unsigned num_threads = min((MAX_BATCH_SIZE / d) * d, N * d);
  unsigned num_blocks = (N * d + num_threads - 1) / num_threads;

  float total_time = 0;
  unsigned int counter = 0;
  while (total_time < 1)
  {
    TIME_START(timing_kernel);
    mergeSmallBatch_k<<<num_blocks, num_threads>>>(v_A_B_gpu, v_A_sizes_gpu, N, d);
    TIME_STOP_SAVE(timing_kernel, time_kernel_ms);
    total_time += time_kernel_ms;
    
    if(total_time<1) cudaMemcpy(v_A_B_gpu,v_A_B_buffer, vector_sizeof(A_B_vectors), cudaMemcpyDeviceToDevice);
    counter++;
  }
  time_kernel_ms = total_time /counter;
  cudaMemcpy(vector_out.data(), v_A_B_gpu, vector_sizeof(A_B_vectors), cudaMemcpyDeviceToHost);

  float time_cpu = merge_batch_cpu(A_B_vectors, A_sizes, N, d);
  bool success = (A_B_vectors == vector_out);
  /* N, d, time_k, time_cpu, success*/
  std::cout << N << ", " << d << ", " << time_kernel_ms << ", " << time_cpu <<", " << success << std::endl;
  cudaFree(v_A_B_gpu);
  cudaFree(v_A_sizes_gpu);
  TIME_EVENT_DESTROY(timing_kernel);

  return EXIT_SUCCESS;
}