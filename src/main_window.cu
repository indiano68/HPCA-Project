#define CUDA_TIMING 

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
           *v_A_B_gpu,
           *v_out_gpu_0;
    int2   *v_Q_gpu;
      
    size_t size_A = std::stoi(argv[1]);
    size_t size_B = std::stoi(argv[2]);

    /*
        Building buffers for that allow the varius benchmakred kernels 
        to store their output
    */
    size_t vector_out_size = size_A + size_B;
    std::vector<v_type> vector_out_0(vector_out_size);
    std::vector<v_type> vector_out_1(vector_out_size);

    float time_0, time_partitioning_squares = 0;
    TIME_EVENT_DEFINE(timing_0);TIME_EVENT_CREATE(timing_0);
    TIME_EVENT_DEFINE(timing_partitioning_squares);TIME_EVENT_CREATE(timing_partitioning_squares);


    if (size_A > size_B)
    {
        std::cout << "Required Size A < Size B!" << std::endl;
        abort();
    }

    dim3 grid_window((vector_out_size + TILE_SIZE * TILES_PER_BLOCK - 1) / (TILE_SIZE * TILES_PER_BLOCK));
    cudaMalloc(&v_A_B_gpu   , vector_out_size * sizeof(v_type));
    v_A_gpu = v_A_B_gpu;
    v_B_gpu = v_A_B_gpu + size_A;
    cudaMalloc(&v_out_gpu_0 , vector_sizeof(vector_out_0));
    cudaMalloc(&v_Q_gpu     , grid_window.x * sizeof(int2));

    generate_random_vectors_cuda(v_A_gpu, size_A, v_B_gpu, size_B, (v_type)-1000, (v_type)1000);

    thrust::sort(thrust::device, v_A_gpu, v_A_gpu + size_A);
    thrust::sort(thrust::device, v_B_gpu, v_B_gpu + size_B);
    cudaDeviceSynchronize();

    /*
    ########################################
        Benchmarking of Tiled Kernel
    ########################################
    */
    //Partitioner computes his grid size from grid size of squares kernel
    dim3 grid_partitioning_window((grid_window.x + THREADS_PER_BLOCK_PARTITIONER - 1) / THREADS_PER_BLOCK_PARTITIONER);
    emptyk<<<1, 1>>>();
    TIME_START(timing_0);TIME_START(timing_partitioning_squares);
    for (int i = 0; i < N_ITER; i++)
    {        

        partition_k_gpu_packed<<<grid_partitioning_window, THREADS_PER_BLOCK_PARTITIONER>>>(v_A_gpu, size_A,
                                                                                            v_B_gpu, size_B,
                                                                                            v_Q_gpu);
        TIME_STOP_SAVE(timing_partitioning_squares, time_partitioning_squares);

        merge_k_gpu_window<<<grid_window, TILE_SIZE>>>(v_A_gpu, size_A,
                                                       v_B_gpu, size_B,
                                                       v_out_gpu_0, v_Q_gpu);


    }
    TIME_STOP_SAVE(timing_0,time_0);

    cudaMemcpy(vector_out_0.data(), v_out_gpu_0, vector_sizeof(vector_out_0), cudaMemcpyDeviceToHost);

    if constexpr(DEBUG) print_vector(vector_out_0);

    float time_3 = bench_thrust_merge(v_A_gpu, size_A, v_B_gpu, size_B, vector_out_1, N_ITER);

    std::cout << "Equality thrust - tiled : " << (vector_out_0 == vector_out_1 ? "True " : "False ") << std::endl;
    std::cout << "Time tiled : " << time_0 / N_ITER << " | Partition T " << time_partitioning_squares / N_ITER << std::endl;
    std::cout << "Time thrust: " << time_3 << std::endl;
    cudaFree(v_A_B_gpu),
    cudaFree(v_out_gpu_0);
    cudaFree(v_Q_gpu);
    TIME_EVENT_DESTROY(timing_0)

    return EXIT_SUCCESS;
}