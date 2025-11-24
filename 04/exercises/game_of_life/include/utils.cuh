#pragma once

#include <limits>
#include <vector>

#define AT(i,j,stride) ((i) + (j) * (stride))

bool float_approx_equal(float a, float b, float epsilon = std::numeric_limits<float>::epsilon());

// Multiplies two matrices on the GPU
template<typename T>
__global__ void matmul_gpu_naive(T* A, T* B, T* O, int i, int k, int j);

// Multiplies two matrices on the CPU
template<typename T>
std::vector<T> matmul_cpu(const std::vector<T>& A, const std::vector<T>& B, int i, int k, int j);

// Check if two matrices are equals or approx equals
template<typename T>
bool mat_check_result(const std::vector<T>& cpu, const std::vector<T>& gpu, int N);

#ifdef PPROG_IMPLEMENTATION

bool float_approx_equal(float a, float b, float epsilon) {
    return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<typename T>
__global__ void matmul_gpu_naive(T* A, T* B, T* O, int i, int k, int j) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T acc = (T)0;
    if (col < j && row < i) {

        for (int e = 0; e < k; ++e) {
            T a = A[row * k + e];
            T b = B[e * j + col];
            acc += a * b;
        }

        O[row * k + col] = acc;
    }
}

template<typename T>
std::vector<T> matmul_cpu(const std::vector<T>& A, const std::vector<T>& B, int i, int k, int j) {

    std::vector<T> result{};
    result.resize(i * j);

    for (int row = 0; row < i; row++) {
        for (int col = 0; col < j; col++) {

            T rval = 0;
            for (int e = 0; e < k; e++) {
                rval += A[row * k + e] * B[e * j + col];
            }
            result[row * j + col] = rval;
        }
    }
    return result;
}

template<typename T>
bool mat_check_result(const std::vector<T>& cpu, const std::vector<T>& gpu, int N) {

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {

            auto a = cpu[y * N + x];
            auto b = gpu[y * N + x];

            if (a != b) {
                printf("invalid element at (%d, %d):\t: A = %d\tB = %d\n",
                    y, x, a, b);
                return false;
            }
        }
    }
    return true;
}

template<>
bool mat_check_result<float>(const std::vector<float>& cpu, const std::vector<float>& gpu, int N) {

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {

            auto a = cpu[y * N + x];
            auto b = gpu[y * N + x];

            if (!float_approx_equal(a, b, 1e-10)) {
                return false;
            }
        }
    }
    return true;
}

#endif