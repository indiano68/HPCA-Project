#include <iostream>
#include <stdio.h>
#include <tools.hpp>
#include <vector>
#include <algorithm>
#include <chrono>
#include <path_merge_tiled.cuh>
#include <wrapper.cuh>
#include <thrust_merge.cuh>
#include <cuda_timing.h>

constexpr int tilesize = 128;
using v_type = int;
#define CUDA_TIMING
int main(int argc, char **argv)
{

    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <size-a> " << " <size-b> " << std::endl;
        abort();
    }
    v_type * d_v_A;
    v_type * d_v_B;
    v_type * d_v_out;
    std::vector<v_type> vector_A = build_random_vector<v_type>(std::stoi(argv[1]), -1000, 1000);
    std::vector<v_type> vector_B = build_random_vector<v_type>(std::stoi(argv[2]), -1000, 1000);
    std::vector<v_type> vector_out(vector_A.size() + vector_B.size());
    std::vector<v_type> vector_out_ref(vector_A.size() + vector_B.size());
    size_t shared_memory_size = (sizeof(v_type)*tilesize*3)+sizeof(int2);
    cudaMalloc(&d_v_A  , vector_sizeof(vector_A));
    cudaMalloc(&d_v_B  , vector_sizeof(vector_B));
    cudaMalloc(&d_v_out, vector_sizeof(vector_out));

    std::sort(vector_A.begin(), vector_A.end());
    std::sort(vector_B.begin(), vector_B.end());
    float time = 0;
    cudaMemcpy(d_v_A, vector_A.data(), vector_sizeof(vector_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_B, vector_B.data(), vector_sizeof(vector_B), cudaMemcpyHostToDevice);
    TIME_EVENT_DEFINE(timing);
    TIME_EVENT_CREATE(timing);
    TIME_START(timing);
    merge_small_tiled_k<<<1,tilesize,shared_memory_size>>>(d_v_A,vector_A.size(),d_v_B,vector_B.size(),d_v_out,vector_out.size());
    TIME_STOP_SAVE(timing,time);
    TIME_EVENT_DESTROY(timing);
    cudaMemcpy(vector_out.data(),d_v_out,vector_sizeof(vector_out),cudaMemcpyDeviceToHost);
    std::merge(vector_A.begin(),vector_A.end(),vector_B.begin(),vector_B.end(),vector_out_ref.begin());
    std::cout << ((vector_out==vector_out_ref) ? "True":"False") <<" " << time <<std::endl;
}
