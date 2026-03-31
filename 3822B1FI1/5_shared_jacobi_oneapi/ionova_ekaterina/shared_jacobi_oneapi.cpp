#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    int n = b.size();
    sycl::queue q(device);

    float* s_a = sycl::malloc_shared<float>(a.size(), q);
    float* s_b = sycl::malloc_shared<float>(n, q);
    float* s_x = sycl::malloc_shared<float>(n, q);
    float* s_x_next = sycl::malloc_shared<float>(n, q);
    float* s_max_diff = sycl::malloc_shared<float>(1, q);

    std::copy(a.begin(), a.end(), s_a);
    std::copy(b.begin(), b.end(), s_b);
    std::fill(s_x, s_x + n, 0.0f);

    for (int k = 0; k < ITERATIONS; ++k) {
        *s_max_diff = 0.0f;

        q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(s_max_diff, sycl::maximum<float>());

            h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> idx, auto& diff) {
                int i = idx[0];
                float sum = 0.0f;
                
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        sum += s_a[i * n + j] * s_x[j];
                    }
                }
                
                s_x_next[i] = (s_b[i] - sum) / s_a[i * n + i];
                
                diff.combine(sycl::fabs(s_x_next[i] - s_x[i]));
            });
        }).wait();

        if (*s_max_diff < accuracy) {
            std::swap(s_x, s_x_next); 
            break;
        }

        std::swap(s_x, s_x_next);
    }

    std::vector<float> result(s_x, s_x + n);

    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_x, q);
    sycl::free(s_x_next, q);
    sycl::free(s_max_diff, q);

    return result;
}