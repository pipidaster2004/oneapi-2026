#include "shared_jacobi_oneapi.h"

#include <algorithm>
#include <vector>

constexpr size_t JACOBI_LOCAL = 64;

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const size_t n = b.size();
    if (n == 0) return {};
    if (a.size() != n * n) return {};
    if (accuracy < 0.0f) accuracy = 0.0f;

    const float accuracy_sq = accuracy * accuracy;

    sycl::queue q(device, sycl::property::queue::in_order{});

    float* A   = sycl::malloc_shared<float>(a.size(), q);
    float* B   = sycl::malloc_shared<float>(b.size(), q);
    float* Inv = sycl::malloc_shared<float>(n, q);
    float* X0  = sycl::malloc_shared<float>(n, q);
    float* X1  = sycl::malloc_shared<float>(n, q);

    float* norm = sycl::malloc_shared<float>(1, q);

    if (!A || !B || !Inv || !X0 || !X1 || !norm) {
        if (A) sycl::free(A, q);
        if (B) sycl::free(B, q);
        if (Inv) sycl::free(Inv, q);
        if (X0) sycl::free(X0, q);
        if (X1) sycl::free(X1, q);
        if (norm) sycl::free(norm, q);
        return {};
    }

    for (size_t i = 0; i < a.size(); ++i) A[i] = a[i];
    for (size_t i = 0; i < b.size(); ++i) B[i] = b[i];

    for (size_t i = 0; i < n; ++i) {
        const float aii = A[i * n + i];
        Inv[i] = (aii != 0.0f) ? (1.0f / aii) : 0.0f;
        X0[i] = 0.0f;
    }

    q.wait_and_throw();

    auto round_up = [](size_t x, size_t m) {
        return ((x + m - 1) / m) * m;
    };
    const size_t global = round_up(n, JACOBI_LOCAL);

    float* x_old = X0;
    float* x_new = X1;

    float norm_host = 0.0f;

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        q.fill(norm, 0.0f, 1);

        sycl::event e = q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(norm, sycl::plus<float>());

            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(JACOBI_LOCAL)),
                red,
                [=](sycl::nd_item<1> it, auto& sum) {
                    const size_t i = it.get_global_id(0);
                    if (i >= n) return;

                    const size_t row = i * n;
                    const float aii = A[row + i];
                    const float inv = Inv[i];

                    float dot = 0.0f;
                    for (size_t j = 0; j < n; ++j) {
                        dot += A[row + j] * x_old[j];
                    }
                    const float sum_excl = dot - aii * x_old[i];

                    const float xi = (B[i] - sum_excl) * inv;
                    x_new[i] = xi;

                    const float d = xi - x_old[i];
                    sum += d * d;
                }
            );
        });

        e.wait_and_throw();

        norm_host = *norm;
        if (norm_host < accuracy_sq) break;

        std::swap(x_old, x_new);
    }

    std::vector<float> result(n);
    for (size_t i = 0; i < n; ++i) result[i] = x_old[i];

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(Inv, q);
    sycl::free(X0, q);
    sycl::free(X1, q);
    sycl::free(norm, q);

    return result;
}