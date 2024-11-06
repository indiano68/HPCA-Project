#include<iostream>
#include<stdio.h>
#include<tools.hpp>
#include<vector>
#include<algorithm>
#include<path_merge.hpp>


int main() 
{
    std::vector<int> int_vector_0    = build_random_vector<int>(10,0,100);
    std::vector<int> int_vector_1    = build_random_vector<int>(10,0,100);
    std::sort(int_vector_0.begin(),int_vector_0.end());
    std::sort(int_vector_1.begin(),int_vector_1.end());
    print_vector(int_vector_0);
    print_vector(int_vector_1);
    auto merged  = path_merge(int_vector_0, int_vector_1);
    print_vector(merged);

}
