#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <path_merge.cuh>
#include <partition.cuh>
#include <thrust_merge.cuh>
#include <utils.hpp>
#include <thrust/sort.h>
#include <randomCudaVector.cuh>

using v_type = float;

int main(int argc, char **argv)
{
    printGPUInfo();
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <size-a> " << " <size-b> " << std::endl;
        abort();
    }

    size_t size_A = std::stoi(argv[1]);
    size_t size_B = std::stoi(argv[2]);
    // if (size_A > size_B)
    // {
    //     std::cout << "Required Size A < Size B!" << std::endl;
    //     abort();
    // }

    std::cout << "Size A: " << size_A << " | Size B: " << size_B << std::endl;
    std::cout << "---------------------------------" << std::endl;

    /*
         Building GPU buffers
    */
    v_type *v_A_gpu, 
           *v_B_gpu,
           *v_A_B_gpu,
           *v_out_gpu;
    int2   *v_Q_gpu_0,
           *v_Q_gpu_1;
      
    /*
        Building buffers for that allow benchmarked kernels 
        to store their output
    */
    size_t vector_out_size = size_A + size_B;
    std::vector<v_type> vector_out(vector_out_size);
    std::vector<v_type> vector_out_thrust(vector_out_size);

    cudaMalloc(&v_A_B_gpu   , vector_out_size * sizeof(v_type));
    v_A_gpu = v_A_B_gpu;
    v_B_gpu = v_A_B_gpu + size_A;
    cudaMalloc(&v_out_gpu , vector_sizeof(vector_out));

    generate_random_vector_gpu(v_A_gpu, size_A);
    generate_random_vector_gpu(v_B_gpu, size_B);
    thrust::sort(thrust::device, v_A_gpu, v_A_gpu + size_A);
    thrust::sort(thrust::device, v_B_gpu, v_B_gpu + size_B);

    /**
     * Benchmarking Thrust Merge
     */
    float time_thrust = bench_thrust_merge(v_A_gpu, size_A, v_B_gpu, size_B, vector_out_thrust);

    /*
    ########################################
        Benchmarking of mergeLarge_window_k
    ########################################
    */
    float time_0, time_partitioning_0 = 0;
    TIME_EVENT_DEFINE(timing_0); TIME_EVENT_DEFINE(timing_partitioning_0);
    {
      TIME_EVENT_CREATE(timing_partitioning_0); TIME_EVENT_CREATE(timing_0);
      dim3 grid_0((vector_out_size + BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK - 1) / (BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK));
      dim3 grid_partitioning_0((grid_0.x + THREADS_PER_BLK_PARTITION - 1) / THREADS_PER_BLK_PARTITION);
      cudaMalloc(&v_Q_gpu_0     , grid_0.x * sizeof(int2));

      emptyk<<<1, 1>>>();
      TIME_START(timing_0);TIME_START(timing_partitioning_0);

      partition_k<<<grid_partitioning_0, THREADS_PER_BLK_PARTITION>>>(v_A_gpu, size_A,
                                                                          v_B_gpu, size_B,
                                                                          v_Q_gpu_0, BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK);
      TIME_STOP_SAVE(timing_partitioning_0, time_partitioning_0);

      mergeLarge_window_k<<<grid_0, BLK_SIZE_WINDOW_K>>>(v_A_gpu, size_A,
                                                      v_B_gpu, size_B,
                                                      v_out_gpu, v_Q_gpu_0);
      TIME_STOP_SAVE(timing_0,time_0);

      cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);

      std::cout << "WINDOW: " << (vector_out == vector_out_thrust ? "PASS" : "FAIL") << std::endl;
    }
    /*
    ########################################
        Benchmarking of mergeLarge_tiled_k
    ########################################
    */
    float time_1, time_partitioning_1 = 0;
    TIME_EVENT_DEFINE(timing_1); TIME_EVENT_DEFINE(timing_partitioning_1);
    {
      TIME_EVENT_CREATE(timing_1); TIME_EVENT_CREATE(timing_partitioning_1);
      dim3 grid_1((vector_out_size + WORK_PER_BLK - 1) / WORK_PER_BLK);
      dim3 grid_partitioning_1((grid_1.x + THREADS_PER_BLK_PARTITION - 1) / THREADS_PER_BLK_PARTITION);
      cudaMalloc(&v_Q_gpu_1     , grid_1.x * sizeof(int2));

      emptyk<<<1, 1>>>();
      TIME_START(timing_1);TIME_START(timing_partitioning_1);
      partition_k<<<grid_partitioning_1, THREADS_PER_BLK_PARTITION>>>(v_A_gpu, size_A,
                                                                          v_B_gpu, size_B,
                                                                          v_Q_gpu_1, WORK_PER_BLK);
      TIME_STOP_SAVE(timing_partitioning_1, time_partitioning_1);

      mergeLarge_tiled_k<<<grid_1, BLK_SIZE_TILED_K>>>(v_A_gpu, size_A,
                                                      v_B_gpu, size_B,
                                                      v_out_gpu, v_Q_gpu_1);
      TIME_STOP_SAVE(timing_1, time_1);

      cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);

      std::cout << "TILED : " << (vector_out == vector_out_thrust ? "PASS" : "FAIL") << std::endl;
    }

    /*
    ########################################
        Benchmarking of mergeLarge_naive_k
    ########################################
    */
    float time_2; TIME_EVENT_DEFINE(timing_2);
    {
      TIME_EVENT_CREATE(timing_2);
      dim3 grid_2((vector_out_size + BLK_SIZE_NAIVE_K - 1) / BLK_SIZE_NAIVE_K);

      emptyk<<<1, 1>>>();
      TIME_START(timing_2);
      mergeLarge_naive_k<<<grid_2, BLK_SIZE_NAIVE_K>>>(v_A_gpu, size_A,
                                                        v_B_gpu, size_B,
                                                        v_out_gpu);
      TIME_STOP_SAVE(timing_2,time_2);

      cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);

      std::cout << "NAIVE : " << (vector_out == vector_out_thrust ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << "---------------------------------" << std::endl;

    std::cout << "Time Thrust: " << time_thrust << std::endl;
    std::cout << "Time window: " << time_0 << " | Partition T " << time_partitioning_0 << std::endl;
    std::cout << "Time tiled : " << time_1 << " | Partition T " << time_partitioning_1 << std::endl;
    std::cout << "Time naive : " << time_2 << std::endl;

    cudaFree(v_A_B_gpu),
    cudaFree(v_out_gpu);
    cudaFree(v_Q_gpu_0);
    cudaFree(v_Q_gpu_1);

    TIME_EVENT_DESTROY(timing_0)
    TIME_EVENT_DESTROY(timing_partitioning_0)
    TIME_EVENT_DESTROY(timing_1)
    TIME_EVENT_DESTROY(timing_partitioning_1)

    return EXIT_SUCCESS;
}