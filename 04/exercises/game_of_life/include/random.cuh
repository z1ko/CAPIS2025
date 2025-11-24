#pragma once

#include <vector>
#include <random>
#include <iterator>
#include <algorithm>

template<typename T>
std::vector<T> random_matrix(int i, int j, unsigned int seed); 

#ifdef PPROG_IMPLEMENTATION

template<>
std::vector<int> random_matrix<int>(int i, int j, unsigned int seed) {

    std::minstd_rand engine(seed);
    std::uniform_int_distribution<int> distr(1, 100);

    std::vector<int> result;
    result.resize(i*j);
    std::generate(result.begin(), result.end(), [&]() {
        return distr(engine) < 35 ? 1 : 0;
    });

    return result;
}

template<>
std::vector<float> random_matrix<float>(int i, int j, unsigned int seed) {
    std::minstd_rand engine(seed);
    std::uniform_real_distribution<float> distr(0.0, 10.0);

    std::vector<float> result;
    result.resize(i*j);
    std::generate(result.begin(), result.end(), [&]() {
        return distr(engine);
    });

    return result;
}

#endif