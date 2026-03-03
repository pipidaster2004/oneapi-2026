#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const int n = static_cast<int>(b.size());
    const float accuracy_sq = accuracy * accuracy;

    sycl::queue queue(device, sycl::property::queue::in_order{});

    std::vector<float> inv_diag(n);
    for (int i = 0; i < n; ++i) {
        inv_diag[i] = 1.0f / a[i * n + i];
    }

    float* d_A = sycl::malloc_device<float>(n * n, queue);
    float* d_B = sycl::malloc_device<float>(n, queue);
    float* d_invD = sycl::malloc_device<float>(n, queue);
    float* d_x = sycl::malloc_device<float>(n, queue);
    float* d_x_new = sycl::malloc_device<float>(n, queue);

    queue.memcpy(d_A, a.data(), sizeof(float) * n * n);
    queue.memcpy(d_B, b.data(), sizeof(float) * n);
    queue.memcpy(d_invD, inv_diag.data(), sizeof(float) * n);
    queue.fill(d_x, 0.0f, n).wait();

    const size_t wg_size = 64;
    const size_t global_size =
        ((n + wg_size - 1) / wg_size) * wg_size;

    const int CHECK_INTERVAL = 8;

    std::vector<float> x_host(n);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.parallel_for(
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> item) {

                size_t i = item.get_global_id(0);
                if (i >= static_cast<size_t>(n)) return;

                const size_t row = i * n;
                float sum = 0.0f;

                #pragma unroll 4
                for (int j = 0; j < n; ++j) {
                    if (j != static_cast<int>(i)) {
                        sum += d_A[row + j] * d_x[j];
                    }
                }

                d_x_new[i] = d_invD[i] * (d_B[i] - sum);
            });

        queue.wait();

        if ((iter + 1) % CHECK_INTERVAL == 0) {

            queue.memcpy(x_host.data(), d_x_new,
                         sizeof(float) * n).wait();

            float norm_sq = 0.0f;

            for (int i = 0; i < n; ++i) {
                float diff = x_host[i];
                norm_sq += diff * diff;
            }

            if (norm_sq < accuracy_sq) {
                std::swap(d_x, d_x_new);
                break;
            }
        }

        std::swap(d_x, d_x_new);
    }

    queue.memcpy(x_host.data(), d_x,
                 sizeof(float) * n).wait();

    sycl::free(d_A, queue);
    sycl::free(d_B, queue);
    sycl::free(d_invD, queue);
    sycl::free(d_x, queue);
    sycl::free(d_x_new, queue);

    return x_host;
}