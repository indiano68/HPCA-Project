# pragma once 

#include <iostream>
#include <vector>
#include <random>
#include <type_traits>

template<class T>
std::vector<T> build_random_vector(size_t size, T min_value, T max_value)
{
    static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");
    std::vector<T> random_vector;
    random_vector.reserve(size);
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(static_cast<double>(min_value), static_cast<double>(max_value));

    for (size_t i = 0; i < size; ++i) {
        random_vector.push_back(static_cast<T>(dis(gen)));
    }
    return random_vector;
}

template<class T>
void print_vector(std::vector<T> vector)
{
    std::cout << "[ ";
    for(auto element: vector)
        std::cout<< element << ", ";
    std::cout<<'\b'<<'\b'; 
    std::cout << " ]"<<std::endl;
}