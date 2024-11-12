# pragma once 

#include <iostream>
#include <vector>
#include <random>
#include <type_traits>

// Add this macro for CUDA error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

template<class T>
std::vector<T> build_random_vector(size_t size, T min_value, T max_value)
{
    static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");
    std::vector<T> random_vector;
    random_vector.reserve(size);
    // Random number generation setup
    std::random_device rd;
    auto seed = rd() + static_cast<unsigned int>(time(nullptr));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(static_cast<double>(min_value), static_cast<double>(max_value));

    for (size_t i = 0; i < size; ++i) {
        random_vector.push_back(static_cast<T>(dis(gen)));
    }
    return random_vector;
}

template<class T>
void print_vector(std::vector<T> vector, const std::string& message = "")
{
    
    if(!message.empty())
        std::cout << message << std::endl;

    std::cout << "[ ";
    for(auto element: vector)
        std::cout<< element << ", ";
    std::cout<<'\b'<<'\b'; 
    std::cout << " ]"<<std::endl;

}

void printGPUInfo()
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  
  printf("Number of devices: %d\n", nDevices);
  
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  }
}