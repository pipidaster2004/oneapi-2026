#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const size_t n = b.size();

    sycl::queue queue(device, sycl::property::queue::in_order{});

    std::vector<float> result(n, 0.0f);

    float* A = sycl::malloc_shared<float>(a.size(), queue);
    float* B = sycl::malloc_shared<float>(b.size(), queue);
    float* x = sycl::malloc_shared<float>(n, queue);
    float* x_new = sycl::malloc_shared<float>(n, queue);
    float* max_diff = sycl::malloc_shared<float>(1, queue);

    queue.memcpy(A, a.data(), a.size() * sizeof(float));
    queue.memcpy(B, b.data(), b.size() * sizeof(float));
    queue.memset(x, 0, sizeof(float) * n);
    queue.memset(x_new, 0, sizeof(float) * n);
    queue.wait();

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        *max_diff = 0.0f;

        auto reduction = sycl::reduction(
            max_diff,
            sycl::maximum<float>());

        queue.parallel_for(
            sycl::range<1>(n),
            reduction,
            [=](sycl::id<1> id, auto& max_red) {

                size_t i = id[0];
                float sum = B[i];
                const size_t row = i * n;

                #pragma unroll 4
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sum -= A[row + j] * x[j];
                    }
                }

                float value = sum / A[row + i];
                x_new[i] = value;

                float diff = sycl::fabs(value - x[i]);
                max_red.combine(diff);
            });

        queue.wait();

        if (*max_diff < accuracy) {
            std::swap(x, x_new);
            break;
        }

        std::swap(x, x_new);
    }

    queue.memcpy(result.data(), x, n * sizeof(float)).wait();

    sycl::free(A, queue);
    sycl::free(B, queue);
    sycl::free(x, queue);
    sycl::free(x_new, queue);
    sycl::free(max_diff, queue);

    return result;
}