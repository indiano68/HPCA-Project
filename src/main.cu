#include<iostream>
#include<stdio.h>
#include<tools.hpp>
#include<vector>
#include<algorithm>
#include<chrono>
#include<path_merge.cuh>
#include <wrapper.cuh>

using v_type = int;

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <size-a> " << " <size-b> " << std::endl;
        abort();
    }
    printGPUInfo();

    std::vector<v_type> vector_1 = build_random_vector<v_type>(std::stoi(argv[1]), -1000, 1000);
    std::vector<v_type> vector_2 = build_random_vector<v_type>(std::stoi(argv[2]), -1000, 1000);
    // std::vector<v_type> vector_1 = {30,50,60,80,110};
    // std::vector<v_type> vector_2 = {10,20,40,70,90,100,120};
    // std::vector<v_type> vector_1 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    // std::vector<v_type> vector_2 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    std::vector<v_type> vector_out0(vector_1.size() + vector_2.size());
    std::vector<v_type> vector_out1(vector_1.size() + vector_2.size());

    v_type *v_1_gpu, *v_2_gpu, *v_out_gpu0, *v_out_gpu1;
    float time0, time1;
    cudaEvent_t start, stop;

    std::sort(vector_1.begin(), vector_1.end());
    std::sort(vector_2.begin(), vector_2.end());

    if(DEBUG)
    {
      std::cout << "Vector 1: " << std::endl;
      print_vector(vector_1);
      std::cout << "Vector 2: " << std::endl;
      print_vector(vector_2);
    }

    
    int block_size = (vector_1.size() + vector_2.size()) / 32;

    cudaMalloc(&v_1_gpu, vector_sizeof(vector_1));
    cudaMalloc(&v_2_gpu, vector_sizeof(vector_2));
    cudaMalloc(&v_out_gpu0, vector_sizeof(vector_out0));
    cudaMalloc(&v_out_gpu1, vector_sizeof(vector_out0));

    empty_k<<<1, 1>>>();
    cudaMemcpy(v_1_gpu, vector_1.data(), vector_sizeof(vector_1), cudaMemcpyHostToDevice);
    cudaMemcpy(v_2_gpu, vector_2.data(), vector_sizeof(vector_2), cudaMemcpyHostToDevice);
    cudaFree(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // for(int i =0; i<100; i++)
    // {
    //     mergeSmall_k2<<<(vector_out0.size() + 1024) / 1024, 1024>>>(v_1_gpu, vector_1.size(),
    //                                                           v_2_gpu, vector_2.size(),
    //                                                           v_out_gpu0, vector_out0.size());
    //     cudaDeviceSynchronize();
    // }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time0, start, stop);

    cudaMemcpy(vector_out0.data(), v_out_gpu0, vector_sizeof(vector_out0), cudaMemcpyDeviceToHost);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // for(int i =0; i<100; i++)
    // {

    //     mergeSmall_k1<<<(vector_out0.size() + 1024) / 1024, 1024>>>(v_1_gpu, vector_1.size(),
    //                                                                 v_2_gpu, vector_2.size(),
    //                                                                 v_out_gpu1, vector_out1.size());
    //     cudaDeviceSynchronize();
    // }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    cudaMemcpy(vector_out1.data(), v_out_gpu1, vector_sizeof(vector_out0), cudaMemcpyDeviceToHost);
    std::cout << "Computed" << std::endl;
    auto merged = mergeSmall_k_cpu(vector_1, vector_2);

    std::cout << "Equality CPU v1: " << (merged == vector_out1 ? "True " : "False ") << "T " << time1 << std::endl;
    std::cout << "Equality CPU v2: " << (merged == vector_out0 ? "True " : "False ") << "T " << time0 << std::endl;

    auto gpu_merge = call_merge_kernel(vector_1, vector_2);

    auto thrust_merge = merge_arrays_thrust(vector_1, vector_2);
    std::cout << thrust_merge.size() << std::endl;

    if(gpu_merge == merged)
    {
        std::cout<<"TEST PASSED!"<<std::endl;
    }
    else
    {
        std::cout<<"GPU and CPU results are not equal"<<std::endl;
    }

    cudaFree(v_1_gpu);
    cudaFree(v_2_gpu);
    cudaFree(v_out_gpu0);
    cudaFree(v_out_gpu1);
    return EXIT_SUCCESS;
}
