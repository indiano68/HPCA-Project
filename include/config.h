#pragma once

int constexpr THREADS_PER_BLOCK = 512;
unsigned constexpr THREADS_PER_BLOCK_PARTITIONER = 32;

//gpu_merge_k_window
constexpr unsigned TILE_SIZE = 512;
constexpr unsigned TILES_PER_BLOCK = 12;
//gpu_merge_k_serial_tile
constexpr unsigned THREADS_PER_BOX = 512; // CUDA block size
constexpr unsigned WORK_PER_THREAD = 15; // number of consecutive elements to process per thread
constexpr unsigned BOX_SIZE = THREADS_PER_BOX * WORK_PER_THREAD; // number of elements to process per block

//enable timing
#define CUDA_TIMING 

constexpr unsigned N_ITER = 1;

typedef int2 coordinate;

int constexpr DEBUG = false;

//test cases
const std::vector<int> A_TEST = {30, 50, 60, 80, 110};
const std::vector<int> B_TEST = {10, 20, 40, 70, 90, 100, 120};

// const std::vector<int> A_TEST = {3,5,6,8,10};
// const std::vector<int> B_TEST = {9,21,41,71,91,101,121};