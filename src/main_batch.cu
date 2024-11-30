#include <iostream>
#include <tools.hpp>
#include <batch_merge_tools.hpp>

using v_type = int;

int main(int argc, char **argv)
{

  if (argc < 3)
  {
      std::cout << "Usage: " << argv[0] << " <N> " << " <d> " << std::endl;
      abort();
  }

  unsigned N = std::stoi(argv[1]);
  unsigned d = std::stoi(argv[2]);
  if(d > 1024 || d < 2)
  {
    std::cout << "d must be between 2 and 1024" << std::endl;
    abort();
  }

  std::vector<v_type> A_B_vectors(N * d);
  std::vector<unsigned> A_offsets = build_and_sort_batches(A_B_vectors, N, d, -10, 10);

  std::vector<v_type> vector_out(N * d);
  v_type *v_A_B_gpu;
  cudaMalloc(&v_A_B_gpu, vector_sizeof(A_B_vectors));
  cudaMemcpy(v_A_B_gpu, A_B_vectors.data(), vector_sizeof(A_B_vectors), cudaMemcpyHostToDevice);

  //kernel call

  cudaMemcpy(vector_out.data(), v_A_B_gpu, vector_sizeof(A_B_vectors), cudaMemcpyDeviceToHost);

  merge_batch_cpu(A_B_vectors, A_offsets, N, d);
  std::cout << "Equality: " << (A_B_vectors == vector_out ? "TRUE" : "FALSE") << std::endl;

  return;
}