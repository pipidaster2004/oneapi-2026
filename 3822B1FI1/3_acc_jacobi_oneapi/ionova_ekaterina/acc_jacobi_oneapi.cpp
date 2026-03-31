#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    int n = b.size();
    std::vector<float> x_host(n, 0.0f);
    std::vector<float> x_next_host(n, 0.0f);

    sycl::queue q(device);

    {
        sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> buf_x(x_host.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> buf_x_next(x_next_host.data(), sycl::range<1>(n));

        for (int k = 0; k < ITERATIONS; ++k) {
            float diff = 0.0f;
            sycl::buffer<float, 1> buf_diff(&diff, 1);

            q.submit([&](sycl::handler& h) {
                auto A = buf_a.get_access<sycl::access::mode::read>(h);
                auto B = buf_b.get_access<sycl::access::mode::read>(h);
                auto X = buf_x.get_access<sycl::access::mode::read>(h);
                auto X_next = buf_x_next.get_access<sycl::access::mode::write>(h);

                auto red = sycl::reduction(buf_diff, h, sycl::maximum<float>());

                h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> idx, auto& max_diff) {
                    int i = idx[0];
                    float sum = 0.0f;
                    
                    for (int j = 0; j < n; ++j) {
                        if (i != j) {
                            sum += A[i * n + j] * X[j];
                        }
                    }
                    
                    X_next[i] = (B[i] - sum) / A[i * n + i];
                    
                    max_diff.combine(sycl::fabs(X_next[i] - X[i]));
                });
            });

            q.wait();
            
            auto acc_diff = buf_diff.get_host_access();
            if (acc_diff[0] < accuracy) {
                auto final_acc = buf_x_next.get_host_access();
                for(int i=0; i<n; ++i) x_host[i] = final_acc[i];
                return x_host;
            }

            std::swap(buf_x, buf_x_next);
        }
        
        auto final_acc = buf_x.get_host_access();
        for(int i=0; i<n; ++i) x_host[i] = final_acc[i];
    }

    return x_host;
}