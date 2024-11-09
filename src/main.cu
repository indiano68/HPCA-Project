#include <iostream>
#include <stdio.h>
#include <tools.hpp>
#include <vector>
#include <algorithm>
#include <path_merge.cuh>
#include <string>


using v_type = int;

int main(int argc, char ** argv)
{

   if(argc<3)
   {
      std::cout << "Usage: " << argv[0] << " <size-a> " << " <size-b> " <<std::endl; 
      abort();
   }
   printGPUInfo();

   std::vector<v_type> vector_1 = build_random_vector<v_type>(std::stoi(argv[1]), -100, 100);
   std::vector<v_type> vector_2 = build_random_vector<v_type>(std::stoi(argv[2]), -100, 100);
   std::vector<v_type> vector_out(vector_1.size() + vector_2.size());
   v_type *v_1_gpu, *v_2_gpu, *v_out_gpu;

   std::sort(vector_1.begin(), vector_1.end());
   std::sort(vector_2.begin(), vector_2.end());

   cudaMalloc(&v_1_gpu  , vector_sizeof(vector_1));
   cudaMalloc(&v_2_gpu  , vector_sizeof(vector_2));
   cudaMalloc(&v_out_gpu, vector_sizeof(vector_out));

   cudaMemcpy(v_1_gpu, vector_1.data(), vector_sizeof(vector_1), cudaMemcpyHostToDevice);
   cudaMemcpy(v_2_gpu, vector_2.data(), vector_sizeof(vector_2), cudaMemcpyHostToDevice);

   mergeSmall_k<<<1, vector_out.size()>>>(v_1_gpu, vector_1.size(),
                                              v_2_gpu, vector_2.size(),
                                              v_out_gpu, vector_out.size());

   cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);

   print_vector(vector_1, "A = ");
   print_vector(vector_2, "B = ");
   auto merged = mergeSmall_k_cpu(vector_1, vector_2);
   print_vector(merged, "CPU merge: ");
   print_vector(vector_out, "GPU merge: ");
}
