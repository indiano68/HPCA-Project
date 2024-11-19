#include <iostream>
#include <stdio.h>
#include <tools.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <path_merge.cuh>
#include <wrapper.cuh>
#include <thrust_merge.cuh>
#include <cuda_timing.h>

using v_type = int;

int constexpr N_ITER = 1;

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
    /*
         Building GPU buffers
    */
    v_type *v_A_gpu, 
           *v_B_gpu, 
           *v_buffer_gpu, 
           *v_out_gpu_0, 
           *v_out_gpu_1,
           *v_out_gpu_2;
    int2   *v_Q_gpu;
    
    /*
        Building vectors to sort 
    */
    std::vector<v_type> vector_A = build_random_vector<v_type>(std::stoi(argv[1]), -1000, 1000);
    std::vector<v_type> vector_B = build_random_vector<v_type>(std::stoi(argv[2]), -1000, 1000);

    /*
        Building buffers for that allow the varius benchmakred kernels 
        to store their output
    */
    int vector_out_size = vector_A.size() + vector_B.size();
    std::vector<v_type> vector_out_0(vector_out_size);
    std::vector<v_type> vector_out_1(vector_out_size);
    std::vector<v_type> vector_out_2(vector_out_size);
    std::vector<v_type> vector_out_3(vector_out_size);
    std::vector<int2> vector_Q;


    float time_0, time_1, time_2, time_3;
    TIME_EVENT_DEFINE(timing_0);TIME_EVENT_CREATE(timing_0);
    TIME_EVENT_DEFINE(timing_1);TIME_EVENT_CREATE(timing_1);
    TIME_EVENT_DEFINE(timing_2);TIME_EVENT_CREATE(timing_2);

    if (vector_A.size() > vector_B.size())
    {
        std::cout << "Required Size A > Size B!" << std::endl;
        abort();
    }

    std::sort(vector_A.begin(), vector_A.end());
    std::sort(vector_B.begin(), vector_B.end());

    if (DEBUG)
    {
        std::cout << "Vector 1: " << std::endl;
        print_vector(vector_A);
        std::cout << "Vector 2: " << std::endl;
        print_vector(vector_B);
    }

    int block_num = (vector_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc(&v_A_gpu     , vector_sizeof(vector_A));
    cudaMalloc(&v_B_gpu     , vector_sizeof(vector_B));
    cudaMalloc(&v_out_gpu_0 , vector_sizeof(vector_out_0));
    cudaMalloc(&v_out_gpu_1 , vector_sizeof(vector_out_1));
    cudaMalloc(&v_out_gpu_2 , vector_sizeof(vector_out_2));
    cudaMalloc(&v_Q_gpu     , (block_num) * sizeof(int2));

    vector_Q.resize(block_num);
    cudaMemcpy(v_A_gpu, vector_A.data(), vector_sizeof(vector_A), cudaMemcpyHostToDevice);
    cudaMemcpy(v_B_gpu, vector_B.data(), vector_sizeof(vector_B), cudaMemcpyHostToDevice);

    emptyk<<<1, 1>>>();

    /*
    ########################################
        Benchmarking of Erik's Kernel
    ########################################
    */

    TIME_START(timing_0);
    for (int i = 0; i < N_ITER; i++)
    {
        partitioner<<<block_num, THREADS_PER_BLOCK>>>(v_A_gpu, vector_A.size(),
                                                      v_B_gpu, vector_B.size(),
                                                      v_Q_gpu, block_num);

        merge_k_blocked<<<block_num, THREADS_PER_BLOCK>>>(v_A_gpu, vector_A.size(),
                                                          v_B_gpu, vector_B.size(),
                                                          v_out_gpu_0, vector_out_0.size(), v_Q_gpu);
    }
    TIME_STOP_SAVE(timing_0,time_0);

    cudaMemcpy(vector_out_0.data(), v_out_gpu_0, vector_sizeof(vector_out_0), cudaMemcpyDeviceToHost);
    cudaMemcpy(vector_Q.data(), v_Q_gpu, vector_sizeof(vector_Q), cudaMemcpyDeviceToHost);

    /* Padding */
    auto remainder = (vector_out_1.size()-1) % THREADS_PER_BLOCK;
    size_t padding = (remainder == 0) ? 0 : THREADS_PER_BLOCK - remainder;
    auto v_buffer = vector_B;
    if (remainder != 0)
    {
        auto biggest_element = std::max(vector_A.back(), vector_B.back());
        v_buffer.resize(v_buffer.size() + padding, biggest_element);
    }
    cudaMalloc(&v_buffer_gpu, vector_sizeof(v_buffer));
    cudaMemcpy(v_buffer_gpu, v_buffer.data(), vector_sizeof(v_buffer), cudaMemcpyHostToDevice);

    /*
    ########################################
        Benchmarking of Triangles Kernel
    ########################################
    */
    TIME_START(timing_1);
    emptyk<<<1, 1>>>();
    for (int i = 0; i < N_ITER; i++)
    {
        merge_k_triangles<<<block_num, THREADS_PER_BLOCK>>>(v_A_gpu, vector_A.size(),
                                                                            v_buffer_gpu, v_buffer.size(),
                                                                            v_out_gpu_1);
    }
    TIME_STOP_SAVE(timing_1,time_1)
    cudaMemcpy(vector_out_1.data(), v_out_gpu_1, vector_sizeof(vector_out_1), cudaMemcpyDeviceToHost);
    /*
    ########################################
        Benchmarking of Squares Kernel
    ########################################
    */
    TIME_START(timing_2);
    emptyk<<<1, 1>>>();
    for (int i = 0; i < N_ITER; i++)
    {
        partition_k_gpu<<<block_num,1>>>(v_A_gpu,vector_A.size(),
                                                         v_buffer_gpu,v_buffer.size(),
                                                         v_Q_gpu);
        merge_k_gpu_squares<<<block_num,THREADS_PER_BLOCK>>>(v_A_gpu,vector_A.size(),
                                                             v_buffer_gpu,v_buffer.size(),
                                                             v_out_gpu_2,v_Q_gpu);
    }
    TIME_STOP_SAVE(timing_2,time_2);

    cudaMemcpy(vector_out_2.data(), v_out_gpu_2, vector_sizeof(vector_out_2), cudaMemcpyDeviceToHost);


    time_3 = bench_thrust_merge(vector_A, vector_B, vector_out_3, N_ITER);
    auto merged = mergeSmall_k_cpu(vector_A, vector_B);
    std::cout << "Equality Erik     mergeLarge    : " << (merged == vector_out_0 ? "True " : "False ") << "T " << time_0 / N_ITER << std::endl;
    std::cout << "Equality Triangle mergeLarge    : " << (merged == vector_out_1 ? "True " : "False ") << "T " << time_1 / N_ITER << std::endl;
    std::cout << "Equality Squares  mergeLarge    : " << (merged == vector_out_1 ? "True " : "False ") << "T " << time_2 / N_ITER << std::endl;
    std::cout << "Equality thrust   merge         : " << (merged == vector_out_3 ? "True " : "False ") << "T " << time_3 << std::endl;
    cudaFree(v_A_gpu), 
    cudaFree(v_B_gpu), 
    cudaFree(v_buffer_gpu), 
    cudaFree(v_out_gpu_0), 
    cudaFree(v_out_gpu_1),
    cudaFree(v_out_gpu_2);
    cudaFree(v_Q_gpu);
    TIME_EVENT_DESTROY(timing_0);
    TIME_EVENT_DESTROY(timing_1);
    TIME_EVENT_DESTROY(timing_2);

    return EXIT_SUCCESS;
}
