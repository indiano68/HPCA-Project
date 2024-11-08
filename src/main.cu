#include<iostream>
#include<stdio.h>
#include<tools.hpp>
#include<vector>
#include<algorithm>
#include<path_merge.cuh>

int main()
{

    printGPUInfo();

    std::vector<double> int_vector_1    = build_random_vector<double>(8,0,100);
    std::vector<double> int_vector_2    = build_random_vector<double>(6,0,100);
    std::vector<double> int_vector_out(int_vector_1.size()+int_vector_2.size());
    double * v_1_gpu, * v_2_gpu, * v_out_gpu;

    std::sort(int_vector_1.begin(),int_vector_1.end());
    std::sort(int_vector_2.begin(),int_vector_2.end());

    cudaMalloc(&v_1_gpu,int_vector_1.size()    *sizeof(double));
    cudaMalloc(&v_2_gpu,int_vector_2.size()    *sizeof(double));
    cudaMalloc(&v_out_gpu,int_vector_out.size()*sizeof(double));

    cudaMemcpy(v_1_gpu,int_vector_1.data(),int_vector_1.size()*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(v_2_gpu,int_vector_2.data(),int_vector_2.size()*sizeof(double),cudaMemcpyHostToDevice);

    mergeSmall_k_gpu<<<1,1>>>(v_1_gpu,int_vector_1.size(),
                              v_2_gpu,int_vector_2.size(),
                              v_out_gpu,int_vector_out.size());

    cudaMemcpy(int_vector_out.data(),v_out_gpu,int_vector_out.size()*sizeof(double),cudaMemcpyDeviceToHost);                         

    print_vector(int_vector_1, "A = ");
    print_vector(int_vector_2, "B = ");
    auto merged  = mergeSmall_k_cpu(int_vector_1, int_vector_2);
    print_vector(merged, "CPU merge: ");
    print_vector(int_vector_out, "GPU merge: ");

}
