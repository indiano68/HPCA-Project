#include <iostream>
#include <stdio.h>
#include <tools.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <path_merge.cuh>
#include <wrapper.cuh>
#include <thrust_merge.cuh>



using v_type = int;

int constexpr N_ITER = 100;

__global__ void emptyk()
{
    return;
}


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
    std::vector<v_type> vector_out0(vector_1.size() + vector_2.size());
    std::vector<v_type> vector_out1(vector_1.size() + vector_2.size());

    v_type *v_1_gpu, *v_2_gpu, *v_buffer_gpu, *v_out_gpu0, * v_out_gpu1;
    float time0, time1;
    cudaEvent_t start, stop;

    if(vector_1.size()>vector_2.size())
    {
        std::cout << "Required Size A > Size B!" << std::endl;
        abort();
    }

    std::sort(vector_1.begin(), vector_1.end());
    std::sort(vector_2.begin(), vector_2.end());
    
    int block_size = (vector_1.size() + vector_2.size()) / 32;

    cudaMalloc(&v_1_gpu, vector_sizeof(vector_1));
    cudaMalloc(&v_2_gpu, vector_sizeof(vector_2));
    cudaMalloc(&v_out_gpu0, vector_sizeof(vector_out0));
    cudaMalloc(&v_out_gpu0, vector_sizeof(vector_out0));

    emptyk<<<1, 1>>>();
    cudaMemcpy(v_1_gpu, vector_1.data(), vector_sizeof(vector_1), cudaMemcpyHostToDevice);
    cudaMemcpy(v_2_gpu, vector_2.data(), vector_sizeof(vector_2), cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    emptyk<<<1, 1>>>();
    for(int i =0; i<N_ITER; i++)
    {

        merge_k_naive<<<(vector_out0.size() + 1024-1) / 1024, 1024>>>(v_1_gpu, vector_1.size(),
                                                                    v_2_gpu, vector_2.size(),
                                                                    v_out_gpu0, vector_out0.size());
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time0, start, stop);
    cudaMemcpy(vector_out0.data(), v_out_gpu0, vector_sizeof(vector_out0), cudaMemcpyDeviceToHost);


    auto remainder = vector_out1.size() % THREADS_PER_BLOCK;
    size_t padding = (remainder == 0) ? 0 : THREADS_PER_BLOCK - remainder;
    auto v_buffer = vector_2;

    if(remainder != 0)
    {
      auto biggest_element = std::max(vector_1.back(), vector_2.back());
      v_buffer.resize(v_buffer.size() + padding, biggest_element);
    }

    cudaMalloc(&v_buffer_gpu, vector_sizeof(v_buffer));
    cudaMalloc(&v_out_gpu1, vector_sizeof(v_buffer)+ vector_sizeof(vector_1));
    cudaMemcpy(v_buffer_gpu, v_buffer.data(), vector_sizeof(v_buffer), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    emptyk<<<1, 1>>>();
    for(int i =0; i<N_ITER; i++)
    {
        
        merge_k_triangles<<<(vector_out1.size() + 1024 - 1) / 1024, 1024>>>(v_1_gpu, vector_1.size(),
                                                                                v_buffer_gpu, v_buffer.size(),
                                                                                v_out_gpu1);
                                
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    cudaMemcpy(vector_out1.data(), v_out_gpu1, vector_sizeof(vector_out1), cudaMemcpyDeviceToHost);
    std::vector<v_type> vector_out2(vector_1.size()+vector_2.size());
    auto time2 = bench_thrust_merge(vector_1,vector_2,vector_out2, N_ITER);
    auto merged = mergeSmall_k_cpu(vector_1, vector_2);
    std::cout << "Equality naive mergeLarge    : " << (merged == vector_out0 ? "True " : "False ") << "T " << time0/N_ITER << std::endl;
    std::cout << "Equality optim mergeLarge v1 : " << (merged == vector_out1 ? "True " : "False ") << "T " << time1/N_ITER << std::endl;
    std::cout << "Equality thrust merge        : " << (merged == vector_out2 ? "True " : "False ") << "T " << time2 << std::endl;
    cudaFree(v_1_gpu);
    cudaFree(v_2_gpu);
    cudaFree(v_buffer_gpu);
    cudaFree(v_out_gpu0);
    cudaFree(v_out_gpu1);
    return EXIT_SUCCESS;
}
