#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    int n = b.size();
    sycl::queue q(device);
    
    float* d_a = sycl::malloc_device<float>(a.size(), q);
    float* d_b = sycl::malloc_device<float>(n, q);
    float* d_x = sycl::malloc_device<float>(n, q);
    float* d_x_next = sycl::malloc_device<float>(n, q);
    
    float* d_max_diff = sycl::malloc_shared<float>(1, q);

    q.memcpy(d_a, a.data(), sizeof(float) * a.size());
    q.memcpy(d_b, b.data(), sizeof(float) * n);
    q.fill(d_x, 0.0f, n);
    q.wait();

    for (int k = 0; k < ITERATIONS; ++k) {
        *d_max_diff = 0.0f;

        q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(d_max_diff, sycl::maximum<float>());

            h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> idx, auto& max_v) {
                int i = idx[0];
                float sum = 0.0f;
                
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        sum += d_a[i * n + j] * d_x[j];
                    }
                }
                
                d_x_next[i] = (d_b[i] - sum) / d_a[i * n + i];
                max_v.combine(sycl::fabs(d_x_next[i] - d_x[i]));
            });
        }).wait();

        if (*d_max_diff < accuracy) {
            std::swap(d_x, d_x_next);
            break;
        }

        std::swap(d_x, d_x_next);
    }

    std::vector<float> result(n);
    q.memcpy(result.data(), d_x, sizeof(float) * n).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_x, q);
    sycl::free(d_x_next, q);
    sycl::free(d_max_diff, q);

    return result;
}