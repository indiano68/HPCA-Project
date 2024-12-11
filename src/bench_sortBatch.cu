#include <iostream>
#include <string>
#include <batch_sort.cuh>
#include <utils.hpp>

#include <cuda_timing.h>

using v_type = int;

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <N> " << " <d> " << std::endl;
        abort();
    }

    unsigned N = std::stoi(argv[1]);
    unsigned d_arg = std::stoi(argv[2]);
    if (d_arg > MAX_BATCH_SIZE || d_arg < 2)
    {
        std::cout << "d must be between 2 and 1024" << std::endl;
        abort();
    }
    else if (d_arg & (d_arg - 1))
    {
        std::cout << "d must be a power of 2" << std::endl;
        abort();
    }

    unsigned short d = static_cast<unsigned short>(d_arg);

    std::vector<v_type> batches(N * d);
    generate_random_vector_cpu(batches, N * d);

    std::vector<v_type> vector_out(N * d);
    v_type *v_M_gpu;
    v_type *v_M_buffer;

    TIME_EVENT_DEFINE(timing_kernel);
    TIME_EVENT_CREATE(timing_kernel);
    float time_kernel_ms = 0.0f;

    cudaMalloc(&v_M_gpu, vector_sizeof(batches));
    cudaMalloc(&v_M_buffer, vector_sizeof(batches));
    cudaMemcpy(v_M_gpu, batches.data(), vector_sizeof(batches), cudaMemcpyHostToDevice);
    cudaMemcpy(v_M_buffer, v_M_gpu, vector_sizeof(batches), cudaMemcpyDeviceToDevice);

    unsigned num_threads = min((MAX_BATCH_SIZE / d) * d, N * d);
    unsigned num_blocks = (N * d + num_threads - 1) / num_threads;
    float total_time = 0;
    unsigned int counter = 0; 

    while(total_time < 1)
    {        
        TIME_START(timing_kernel);
        sortSmallBatch_k<<<num_blocks, num_threads>>>(v_M_gpu, N, d);
        TIME_STOP_SAVE(timing_kernel, time_kernel_ms);
        total_time+=time_kernel_ms;
        if(total_time< 1) cudaMemcpy(v_M_gpu,v_M_buffer, vector_sizeof(batches), cudaMemcpyDeviceToDevice);
        counter++;
    }
    cudaMemcpy(vector_out.data(), v_M_gpu, vector_sizeof(batches), cudaMemcpyDeviceToHost);
    time_kernel_ms = total_time/counter;
    float time_cpu = sort_batch_cpu(batches, N, d);
    bool success = (batches == vector_out);
    std::cout << N << ", " << d << ", " << time_kernel_ms << ", " << time_cpu <<", " << success << std::endl;
    cudaFree(v_M_gpu);
    TIME_EVENT_DESTROY(timing_kernel);

    return EXIT_SUCCESS;
}