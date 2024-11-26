#pragma once

int constexpr THREADS_PER_BLOCK = 512;
unsigned constexpr THREADS_PER_BLOCK_PARTITIONER = 32;

constexpr unsigned TILE_SIZE = 512;
constexpr unsigned TILES_PER_BLOCK = 12;

//enable timing
#define CUDA_TIMING 

constexpr unsigned N_ITER = 1;

typedef int2 coordinate;

int constexpr DEBUG = false;