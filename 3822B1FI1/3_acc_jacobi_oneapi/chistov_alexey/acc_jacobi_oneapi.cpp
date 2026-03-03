#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {

    const size_t n = static_cast<size_t>(std::sqrt(a.size()));
    const float accuracy_sq = accuracy * accuracy;

    std::vector<float> inv_diag(n);
    for (size_t i = 0; i < n; ++i) {
        inv_diag[i] = 1.0f / a[i * n + i];
    }

    sycl::queue queue(device, sycl::property::queue::in_order{});

    sycl::buffer<float> A_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> B_buf(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float> inv_diag_buf(inv_diag.data(), sycl::range<1>(n));

    sycl::buffer<float> x_buf(sycl::range<1>(n));
    sycl::buffer<float> x_new_buf(sycl::range<1>(n));
    sycl::buffer<float> diff_norm_buf(sycl::range<1>(1));

    queue.submit([&](sycl::handler& cgh) {
        auto x = x_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.fill(x, 0.0f);
    });

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        queue.submit([&](sycl::handler& cgh) {
            auto A = A_buf.get_access<sycl::access::mode::read>(cgh);
            auto B = B_buf.get_access<sycl::access::mode::read>(cgh);
            auto invD = inv_diag_buf.get_access<sycl::access::mode::read>(cgh);
            auto x = x_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_new = x_new_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                size_t i = id[0];
                size_t row = i * n;

                float sum = 0.0f;

                #pragma unroll 4
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += A[row + j] * x[j];
                    }
                }

                x_new[i] = invD[i] * (B[i] - sum);
            });
        });

        queue.submit([&](sycl::handler& cgh) {
            auto x = x_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_new = x_new_buf.get_access<sycl::access::mode::read>(cgh);

            auto reduction = sycl::reduction(
                diff_norm_buf,
                cgh,
                0.0f,
                sycl::plus<float>());

            cgh.parallel_for(sycl::range<1>(n), reduction,
                [=](sycl::id<1> id, auto& sum) {
                    float diff = x_new[id] - x[id];
                    sum += diff * diff;
                });
        }).wait();

        float norm_sq = diff_norm_buf.get_host_access()[0];

        if (norm_sq < accuracy_sq) {
            break;
        }

        std::swap(x_buf, x_new_buf);
    }

    std::vector<float> result(n);

    queue.submit([&](sycl::handler& cgh) {
        auto x = x_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(x, result.data());
    }).wait();

    return result;
}