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
#include <string>

using v_type = float;

int main(int argc, char **argv)
{

    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <size-a> " << " <size-b> " << " <kernel> " << std::endl;
        abort();
    }
    size_t size_A = std::stoi(argv[1]);
    size_t size_B = std::stoi(argv[2]);
    std::string kernel = argv[3];
    if (kernel != "tiled_k" && kernel != "window_k" && kernel != "naive_k" && kernel != "thrust")
    {
        std::cout << "Kernel: '" << kernel << "' not found!" << std::endl;
        abort();
    }

    v_type *v_A_gpu,
        *v_B_gpu,
        *v_A_B_gpu,
        *v_out_gpu;
    int2 *v_Q_gpu_0,
        *v_Q_gpu_1;

    size_t vector_out_size = size_A + size_B;
    std::vector<v_type> vector_out(vector_out_size);
    std::vector<v_type> vector_out_thrust(vector_out_size);
    cudaMalloc(&v_A_B_gpu, vector_out_size * sizeof(v_type));
    v_A_gpu = v_A_B_gpu;
    v_B_gpu = v_A_B_gpu + size_A;
    cudaMalloc(&v_out_gpu, vector_sizeof(vector_out));
    generate_random_vector_gpu(v_A_gpu, size_A);
    generate_random_vector_gpu(v_B_gpu, size_B);
    thrust::sort(thrust::device, v_A_gpu, v_A_gpu + size_A);
    thrust::sort(thrust::device, v_B_gpu, v_B_gpu + size_B);

    float time_thrust = bench_thrust_merge(v_A_gpu, size_A, v_B_gpu, size_B, vector_out_thrust);

    float time_0 = 0, time_total = 0;
    unsigned int counter= 0;
    bool success = false;
    TIME_EVENT_DEFINE(timing_0);
    TIME_EVENT_CREATE(timing_0);
    if (kernel == "tiled_k")
    {
        dim3 grid_1((vector_out_size + WORK_PER_BLK - 1) / WORK_PER_BLK);
        dim3 grid_partitioning_1((grid_1.x + THREADS_PER_BLK_PARTITION - 1) / THREADS_PER_BLK_PARTITION);
        cudaMalloc(&v_Q_gpu_1, grid_1.x * sizeof(int2));
        partition_k<<<grid_partitioning_1, THREADS_PER_BLK_PARTITION>>>(v_A_gpu, size_A,
                                                                        v_B_gpu, size_B,
                                                                        v_Q_gpu_1, WORK_PER_BLK);
        mergeLarge_tiled_k<<<grid_1, BLK_SIZE_TILED_K>>>(v_A_gpu, size_A,
                                                         v_B_gpu, size_B,
                                                         v_out_gpu, v_Q_gpu_1);
        while(time_total<1)
        {                                                         
            TIME_START(timing_0);
            partition_k<<<grid_partitioning_1, THREADS_PER_BLK_PARTITION>>>(v_A_gpu, size_A,
                                                                            v_B_gpu, size_B,
                                                                            v_Q_gpu_1, WORK_PER_BLK);
            mergeLarge_tiled_k<<<grid_1, BLK_SIZE_TILED_K>>>(v_A_gpu, size_A,
                                                            v_B_gpu, size_B,
                                                            v_out_gpu, v_Q_gpu_1);

            TIME_STOP_SAVE(timing_0, time_0);
            time_total+=time_0;
            counter++;
        }
        time_0 =time_total/counter;
        cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);
        success = (vector_out == vector_out_thrust);
        cudaFree(v_Q_gpu_1);
    }
    else if (kernel == "window_k")
    {
        dim3 grid_0((vector_out_size + BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK - 1) / (BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK));
        dim3 grid_partitioning_0((grid_0.x + THREADS_PER_BLK_PARTITION - 1) / THREADS_PER_BLK_PARTITION);
        cudaMalloc(&v_Q_gpu_0, grid_0.x * sizeof(int2));
        // WARMPUP_CALLS
        partition_k<<<grid_partitioning_0, THREADS_PER_BLK_PARTITION>>>(v_A_gpu, size_A,
                                                                        v_B_gpu, size_B,
                                                                        v_Q_gpu_0, BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK);
        mergeLarge_window_k<<<grid_0, BLK_SIZE_WINDOW_K>>>(v_A_gpu, size_A,
                                                           v_B_gpu, size_B,
                                                           v_out_gpu, v_Q_gpu_0);
        while(time_total<1)
        {                                                         

            TIME_START(timing_0);
            partition_k<<<grid_partitioning_0, THREADS_PER_BLK_PARTITION>>>(v_A_gpu, size_A,
                                                                            v_B_gpu, size_B,
                                                                            v_Q_gpu_0, BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK);
            mergeLarge_window_k<<<grid_0, BLK_SIZE_WINDOW_K>>>(v_A_gpu, size_A,
                                                            v_B_gpu, size_B,
                                                            v_out_gpu, v_Q_gpu_0);
            TIME_STOP_SAVE(timing_0, time_0);

            time_total+=time_0;
            counter++;
        }
        time_0 =time_total/counter;
        cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);
        success = (vector_out == vector_out_thrust);
        cudaFree(v_Q_gpu_0);
    }
    else if (kernel == "naive_k")
    {
        dim3 grid_2((vector_out_size + BLK_SIZE_NAIVE_K - 1) / BLK_SIZE_NAIVE_K);
        /* WARMUP_CALL*/
        mergeLarge_naive_k<<<grid_2, BLK_SIZE_NAIVE_K>>>(v_A_gpu, size_A,
                                                         v_B_gpu, size_B,
                                                         v_out_gpu);
        while(time_total<1)
        {                                                         

            TIME_START(timing_0);
            mergeLarge_naive_k<<<grid_2, BLK_SIZE_NAIVE_K>>>(v_A_gpu, size_A,
                                                                v_B_gpu, size_B,
                                                                v_out_gpu);
            TIME_STOP_SAVE(timing_0, time_0);
            time_total+=time_0;
            counter++;
        }
        time_0 =time_total/counter;
        cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);
        success = (vector_out == vector_out_thrust);
    }
    else if (kernel == "thrust")
    {
        while(time_total<1)
        {                                                         
            time_0 = bench_thrust_merge(v_A_gpu, size_A, v_B_gpu, size_B, vector_out_thrust);
            time_total+=time_0;
            counter++;
        }        
        time_0 =time_total/counter;
        success = true;
    }
    /* total_size, A_size, B_size, time */
    std::cout <<vector_out_size<< ", "<< size_A << ", "<< size_B<< ", " << time_0 << ", " << success << std::endl;
    cudaFree(v_A_B_gpu);
    cudaFree(v_out_gpu);
    TIME_EVENT_DESTROY(timing_0);
}