#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using std::numeric_limits;

template <class T>
const std::vector<unsigned short> build_and_sort_batches(std::vector<T> &batches, unsigned N, unsigned short d, T min = numeric_limits<T>::min(), T max = numeric_limits<T>::max())
{
  std::cout << "Generating and sorting " << N << " random batches of size " << d << " ..." << std::endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min, max);
  std::generate(batches.begin(), batches.end(), [&dis, &gen]() { return static_cast<T>(dis(gen)); });

  std::uniform_int_distribution<unsigned short> splitter(1, d - 1);
  std::vector<unsigned short> offsets(N);
  std::generate(offsets.begin(), offsets.end(), [&splitter, &gen]() { return splitter(gen); });

#pragma omp parallel for
  for(unsigned batch_idx = 0; batch_idx < N; batch_idx++)
  {
    auto curr_batch_start = batches.begin() + batch_idx * d;
    std::sort(curr_batch_start, curr_batch_start + offsets[batch_idx]);
    std::sort(curr_batch_start + offsets[batch_idx], curr_batch_start + d);
  }
  std::cout << "Batches sorted." << std::endl;

  return offsets;
}

template <class T>
void merge_batch_cpu(std::vector<T> &batches, const std::vector<unsigned short> &offsets, unsigned N, unsigned short d)
{
  auto start = std::chrono::high_resolution_clock::now();
  for(unsigned batch_idx = 0; batch_idx < N; batch_idx++)
  {
    auto curr_batch_start = batches.begin() + batch_idx * d;
    std::inplace_merge(curr_batch_start, curr_batch_start + offsets[batch_idx], curr_batch_start + d);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU merge time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

template <class T>
void print_vec_batch(std::vector<T> &batches, const std::vector<unsigned short> &offsets, unsigned N, unsigned short d)
{
  for(unsigned batch_idx = 0; batch_idx < N; batch_idx++)
  {
    std::cout << "Batch " << batch_idx << ": ";
    auto curr_batch_start = batches.begin() + batch_idx * d;
    std::cout << "A: ";
    for(auto i = 0; i < offsets[batch_idx]; i++)
    {
      std::cout << curr_batch_start[i] << " ";
    }
    std::cout << "B: ";
    for(auto i = offsets[batch_idx]; i < d; i++)
    {
      std::cout << curr_batch_start[i] << " ";
    }
    std::cout << std::endl;
  }
}