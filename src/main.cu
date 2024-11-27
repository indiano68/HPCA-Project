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
           *v_out_gpu_0, 
           *v_out_gpu_1;
    int2   *v_Q_gpu,
           *v_Q_gpu_window,
            *v_Q_gpu_serial_tile;
    
    /*
        Building vectors to sort 
    */
    std::vector<v_type> vector_A = build_random_vector<v_type>(std::stoi(argv[1]));
    std::vector<v_type> vector_B = build_random_vector<v_type>(std::stoi(argv[2]));

    // std::vector<v_type> vector_A = A_TEST;
    // std::vector<v_type> vector_B = B_TEST;

    /*
        Building buffers for that allow the varius benchmakred kernels 
        to store their output
    */
    int vector_out_size = vector_A.size() + vector_B.size();
    std::vector<v_type> vector_out_0(vector_out_size);
    std::vector<v_type> vector_out_1(vector_out_size);
    std::vector<v_type> vector_out_2(vector_out_size);
    std::vector<int2> vector_Q;


    float time_erik = 0, time_window = 0, time_thrust = 0, time_partitioning_erik = 0, time_partitioning_window = 0;
    TIME_EVENT_DEFINE(timing_erik);TIME_EVENT_CREATE(timing_erik);
    TIME_EVENT_DEFINE(timing_window);TIME_EVENT_CREATE(timing_window);
    TIME_EVENT_DEFINE(timing_partitioning_erik);TIME_EVENT_CREATE(timing_partitioning_erik);
    TIME_EVENT_DEFINE(timing_partitioning_window);TIME_EVENT_CREATE(timing_partitioning_window);

    if (vector_A.size() > vector_B.size())
    {
        std::cout << "Required Size A < Size B!" << std::endl;
        abort();
    }

    std::sort(vector_A.begin(), vector_A.end());
    std::sort(vector_B.begin(), vector_B.end());

    if constexpr(DEBUG)
    {
        std::cout << "Vector A: " << std::endl;
        print_vector(vector_A);
        std::cout << "Vector B: " << std::endl;
        print_vector(vector_B);
    }

    int block_num = (vector_out_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid_window((vector_out_size + TILE_SIZE * TILES_PER_BLOCK - 1) / (TILE_SIZE * TILES_PER_BLOCK));
    dim3 grid_serial_tile((vector_out_size + BOX_SIZE - 1) / BOX_SIZE);

    cudaMalloc(&v_A_B_gpu   , vector_sizeof(vector_A) + vector_sizeof(vector_B));
    v_A_gpu = v_A_B_gpu;
    v_B_gpu = v_A_B_gpu + vector_A.size();

    cudaMalloc(&v_out_gpu_0 , vector_sizeof(vector_out_0));
    cudaMalloc(&v_out_gpu_1 , vector_sizeof(vector_out_1));

    cudaMalloc(&v_Q_gpu     , (block_num) * sizeof(int2));
    cudaMalloc(&v_Q_gpu_window, grid_window.x * sizeof(int2));
    cudaMalloc(&v_Q_gpu_serial_tile, grid_serial_tile.x * sizeof(int2));

    vector_Q.resize(block_num);
    cudaMemcpy(v_A_gpu, vector_A.data(), vector_sizeof(vector_A), cudaMemcpyHostToDevice);
    cudaMemcpy(v_B_gpu, vector_B.data(), vector_sizeof(vector_B), cudaMemcpyHostToDevice);

    /*
    ########################################
        Benchmarking of Erik's Kernel
    ########################################
    */
    emptyk<<<1, 1>>>();
    TIME_START(timing_erik);TIME_START(timing_partitioning_erik);
    for (int i = 0; i < N_ITER; i++)
    {
        partitioner<<<block_num, 1>>>(v_A_gpu, vector_A.size(),
                                      v_B_gpu, vector_B.size(),
                                      v_Q_gpu, block_num);

        TIME_STOP_SAVE(timing_partitioning_erik, time_partitioning_erik);

        merge_k_blocked<<<block_num, THREADS_PER_BLOCK>>>(v_A_gpu, vector_A.size(),
                                                          v_B_gpu, vector_B.size(),
                                                          v_out_gpu_0, vector_out_0.size(), v_Q_gpu);
    }
    TIME_STOP_SAVE(timing_erik,time_erik);

    cudaMemcpy(vector_out_0.data(), v_out_gpu_0, vector_sizeof(vector_out_0), cudaMemcpyDeviceToHost);
    cudaMemcpy(vector_Q.data(), v_Q_gpu, vector_sizeof(vector_Q), cudaMemcpyDeviceToHost);
    
    /*
    ########################################
        Benchmarking of Tiled Kernel
    ########################################
    */
    dim3 grid_partitioning_window((grid_window.x + THREADS_PER_BLOCK_PARTITIONER - 1) / THREADS_PER_BLOCK_PARTITIONER);
    dim3 grid_partitioning_serial_tile((grid_serial_tile.x + THREADS_PER_BLOCK_PARTITIONER - 1) / THREADS_PER_BLOCK_PARTITIONER);
    emptyk<<<1, 1>>>();
    TIME_START(timing_window);TIME_START(timing_partitioning_window);
    for (int i = 0; i < N_ITER; i++)
    {        
        // partition_k_gpu_packed<<<grid_partitioning_window, THREADS_PER_BLOCK_PARTITIONER>>>(v_A_gpu, vector_A.size(),
        //                                                                                     v_B_gpu, vector_B.size(),
        //                                                                                     v_Q_gpu_window);
        // TIME_STOP_SAVE(timing_partitioning_window, time_partitioning_window);

        // merge_k_gpu_window<<<grid_window, TILE_SIZE>>>(v_A_gpu, vector_A.size(),
        //                                                v_B_gpu, vector_B.size(),
        //                                                v_out_gpu_1, v_Q_gpu_window);

        partition_k_gpu_packed<<<grid_partitioning_serial_tile, THREADS_PER_BLOCK_PARTITIONER>>>(v_A_gpu, vector_A.size(),
                                                                                                 v_B_gpu, vector_B.size(),
                                                                                                 v_Q_gpu_serial_tile, BOX_SIZE);
        TIME_STOP_SAVE(timing_partitioning_window, time_partitioning_window);

        merge_k_gpu_serial_tile_shared<<<grid_serial_tile, THREADS_PER_BOX>>>(v_A_gpu, vector_A.size(),
                                                                       v_B_gpu, vector_B.size(),
                                                                       v_out_gpu_1, v_Q_gpu_serial_tile);
    }
    TIME_STOP_SAVE(timing_window,time_window);

    cudaMemcpy(vector_out_1.data(), v_out_gpu_1, vector_sizeof(vector_out_1), cudaMemcpyDeviceToHost);

    if constexpr(DEBUG) print_vector(vector_out_1);

    time_thrust = bench_thrust_merge(vector_A, vector_B, vector_out_2, N_ITER);
    auto merged = mergeSmall_k_cpu(vector_A, vector_B);

    std::cout << "Equality Erik     mergeLarge    : " << (merged == vector_out_0 ? "True " : "False ") << "T " << time_erik / N_ITER << " | Partition T " << time_partitioning_erik / N_ITER << std::endl;
    std::cout << "Equality Tiled  mergeLarge    : " << (merged == vector_out_1 ? "True " : "False ") << "T " << time_window / N_ITER << " | Partition T " << time_partitioning_window / N_ITER << std::endl;
    std::cout << "Equality thrust   merge         : " << (merged == vector_out_2 ? "True " : "False ") << "T " << time_thrust << std::endl;

    cudaFree(v_A_B_gpu),
    cudaFree(v_out_gpu_0), 
    cudaFree(v_out_gpu_1);
    cudaFree(v_Q_gpu);

    TIME_EVENT_DESTROY(timing_erik);
    TIME_EVENT_DESTROY(timing_partitioning_erik);
    TIME_EVENT_DESTROY(timing_window);
    TIME_EVENT_DESTROY(timing_partitioning_window);

    return EXIT_SUCCESS;
}
