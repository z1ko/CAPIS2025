// compute-sanitizer --tool memcheck

#include <iostream>

__device__ void fn2(int k, int p)
{
    *(int*) k = p;
}

__global__ void fn1(int p)
{
    fn2(0x87654320, p % 32);
}

static void fn0(void)
{
    std::cout << "Running kernel fn0\n";
    fn1<<<1,1>>>(32147);
    cudaDeviceSynchronize();
}

int main() {
    fn0();
    return 0;
}
