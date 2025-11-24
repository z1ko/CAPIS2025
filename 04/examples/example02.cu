// compute-sanitizer --leak-check full --tool memcheck

#include <iostream>

__device__ int x;

__global__ void fn1(int p)
{
    *(int*) ((char*)&x + 1) = p * 2;
}

static void fn0(void)
{
    fn1<<<1,1>>>(42);
    cudaDeviceSynchronize();
}

int main() {

    int *devMem = nullptr;
    cudaMalloc((void**)&devMem, 1024);
    fn0();
    
    return 0;
}
