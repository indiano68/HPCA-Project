#include<iostream>
#include<stdio.h>


__global__ void  hello_from_gpu()
{
    printf("Hello World from GPU: Th %d  Blk %d\n",threadIdx.x,blockIdx.x); 
}

int main() 
{
    std::cout<< "Hello World" <<std::endl;
    hello_from_gpu<<<1,32>>>();
}
