#ifndef __ACC_JACOBI_ONEAPI_H
#define __ACC_JACOBI_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device);

#endif  // __ACC_JACOBI_ONEAPI_H
