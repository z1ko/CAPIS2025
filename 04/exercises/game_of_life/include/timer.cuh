#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

class timerGPU {
public:
    timerGPU() = default;
    ~timerGPU();

    void start();
    void stop();

    float elapsed_ms();

private:
    cudaEvent_t m_start, m_end;
};

class timerCPU {
public:
    void start();
    void stop();

    float elapsed_ms();

private:
    timespec m_start, m_end;
};

#ifdef PPROG_IMPLEMENTATION

timerGPU::~timerGPU() {
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_end);
}

void timerGPU::start() {
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_end);
    cudaEventRecord(m_start, 0);
}

void timerGPU::stop() {
    cudaEventRecord(m_end, 0);
    cudaEventSynchronize(m_end);
}

float timerGPU::elapsed_ms() {
    float ms;
    cudaEventElapsedTime(&ms, m_start, m_end);
    return ms;
}

void timerCPU::start() {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &m_start);
}

void timerCPU::stop() {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &m_end);
}

float timerCPU::elapsed_ms() {
    long ns = (m_end.tv_sec - m_start.tv_sec) * (long)1e9 + (m_end.tv_nsec - m_start.tv_nsec);
    return ns * 1e-6;
}

#endif