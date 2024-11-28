#include <iostream>
#include <stdio.h>
#include <tools.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <path_merge.cuh>
#include <partition.cuh>
#include <wrapper.cuh>
#include <thrust_merge.cuh>
#include <randomCudaVector.cuh>

using v_type = int;

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
           *v_A_B_gpu,
           *v_out_gpu;
    int2   *v_Q_gpu_0,
           *v_Q_gpu_1;
      
    size_t size_A = std::stoi(argv[1]);
    size_t size_B = std::stoi(argv[2]);

    std::cout << "Size A: " << size_A << " | Size B: " << size_B << std::endl;
    std::cout << "---------------------------------" << std::endl;

    /*
        Building buffers for that allow the varius benchmakred kernels 
        to store their output
    */
    size_t vector_out_size = size_A + size_B;
    std::vector<v_type> vector_out(vector_out_size);
    std::vector<v_type> vector_out_thrust(vector_out_size);

    float time_0, time_partitioning_0 = 0;
    TIME_EVENT_DEFINE(timing_0);TIME_EVENT_CREATE(timing_0);
    TIME_EVENT_DEFINE(timing_partitioning_0);TIME_EVENT_CREATE(timing_partitioning_0);

    float time_1, time_partitioning_1 = 0;
    TIME_EVENT_DEFINE(timing_1);TIME_EVENT_CREATE(timing_1);
    TIME_EVENT_DEFINE(timing_partitioning_1);TIME_EVENT_CREATE(timing_partitioning_1);


    if (size_A > size_B)
    {
        std::cout << "Required Size A < Size B!" << std::endl;
        abort();
    }

    dim3 grid_0((vector_out_size + TILE_SIZE * TILES_PER_BLOCK - 1) / (TILE_SIZE * TILES_PER_BLOCK));
    dim3 grid_1((vector_out_size + BOX_SIZE - 1) / BOX_SIZE);

    dim3 grid_partitioning_0((grid_0.x + THREADS_PER_BLOCK_PARTITIONER - 1) / THREADS_PER_BLOCK_PARTITIONER);
    dim3 grid_partitioning_1((grid_1.x + THREADS_PER_BLOCK_PARTITIONER - 1) / THREADS_PER_BLOCK_PARTITIONER);

    cudaMalloc(&v_A_B_gpu   , vector_out_size * sizeof(v_type));
    v_A_gpu = v_A_B_gpu;
    v_B_gpu = v_A_B_gpu + size_A;

    cudaMalloc(&v_out_gpu , vector_sizeof(vector_out));

    cudaMalloc(&v_Q_gpu_0     , grid_0.x * sizeof(int2));
    cudaMalloc(&v_Q_gpu_1     , grid_1.x * sizeof(int2));

    generate_random_vectors_cuda(v_A_gpu, size_A, v_B_gpu, size_B);

    thrust::sort(thrust::device, v_A_gpu, v_A_gpu + size_A);
    thrust::sort(thrust::device, v_B_gpu, v_B_gpu + size_B);

    float time_thrust = bench_thrust_merge(v_A_gpu, size_A, v_B_gpu, size_B, vector_out_thrust, N_ITER);

    /*
    ########################################
        Benchmarking of Tiled Kernel
    ########################################
    */
    emptyk<<<1, 1>>>();
    TIME_START(timing_0);TIME_START(timing_partitioning_0);
    for (int i = 0; i < N_ITER; i++)
    {        

        partition_k_gpu_packed<<<grid_partitioning_0, THREADS_PER_BLOCK_PARTITIONER>>>(v_A_gpu, size_A,
                                                                                            v_B_gpu, size_B,
                                                                                            v_Q_gpu_0);
        TIME_STOP_SAVE(timing_partitioning_0, time_partitioning_0);

        merge_k_gpu_window<<<grid_0, TILE_SIZE>>>(v_A_gpu, size_A,
                                                       v_B_gpu, size_B,
                                                       v_out_gpu, v_Q_gpu_0);
    }
    TIME_STOP_SAVE(timing_0,time_0);

    cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);

    std::cout << "WINDOW: " << (vector_out == vector_out_thrust ? "PASS" : "FAIL") << std::endl;
    
    /*
    ########################################
        Benchmarking of serial_tile Kernel
    ########################################
    */
    emptyk<<<1, 1>>>();
    TIME_START(timing_1);TIME_START(timing_partitioning_1);
    for (int i = 0; i < N_ITER; i++)
    {
        partition_k_gpu_packed<<<grid_partitioning_1, THREADS_PER_BLOCK_PARTITIONER>>>(v_A_gpu, size_A,
                                                                                       v_B_gpu, size_B,
                                                                                       v_Q_gpu_1, BOX_SIZE);
        TIME_STOP_SAVE(timing_partitioning_1, time_partitioning_1);

        merge_k_gpu_serial_tiled<<<grid_1, THREADS_PER_BOX>>>(v_A_gpu, size_A,
                                                                              v_B_gpu, size_B,
                                                                              v_out_gpu, v_Q_gpu_1);
    }
    TIME_STOP_SAVE(timing_1,time_1);

    cudaMemcpy(vector_out.data(), v_out_gpu, vector_sizeof(vector_out), cudaMemcpyDeviceToHost);

    std::cout << "SERIAL TILE: " << (vector_out == vector_out_thrust ? "PASS" : "FAIL") << std::endl;
    std::cout << "---------------------------------" << std::endl;

    std::cout << "Time Thrust:      " << time_thrust << std::endl;
    std::cout << "Time window:      " << time_0 / N_ITER << " | Partition T " << time_partitioning_0 / N_ITER << std::endl;
    std::cout << "Time serial_tile: " << time_1 / N_ITER << " | Partition T " << time_partitioning_1 / N_ITER << std::endl;


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