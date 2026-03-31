#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    
    size_t n = b.size(); 
    std::vector<float> x_curr(n, 0.0f); 
    std::vector<float> x_next(n, 0.0f); 
    
    try {
        sycl::queue queue(device);
        sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(n * n));
        sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> buf_x_curr(x_curr.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> buf_x_next(x_next.data(), sycl::range<1>(n));
        
        bool converged = false;
        
        for (int iter = 0; iter < ITERATIONS && !converged; iter++) {
            queue.submit([&](sycl::handler& h) {
                auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
                auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
                auto acc_x_curr = buf_x_curr.get_access<sycl::access::mode::read>(h);
                auto acc_x_next = buf_x_next.get_access<sycl::access::mode::write>(h);
                
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    size_t row = i[0];
                    float sum = 0.0f;
                    float a_ii = 0.0f;
   			a_ij * x_j для j != i
                    for (size_t j = 0; j < n; j++) {
                        if (j != row) {
                            sum += acc_a[row * n + j] * acc_x_curr[j];
                        } else {
                            a_ii = acc_a[row * n + row];
                        }
                    }
        
                    if (std::abs(a_ii) > 1e-10f) { 
                        acc_x_next[row] = (acc_b[row] - sum) / a_ii;
                    } else {
                        acc_x_next[row] = 0.0f;
                    }
                });
            }).wait();
            sycl::buffer<float, 1> buf_max_diff(sycl::range<1>(1));

            {
                auto host_acc = buf_max_diff.get_host_access();
                host_acc[0] = 0.0f;
            }
            
            queue.submit([&](sycl::handler& h) {
                auto acc_x_curr = buf_x_curr.get_access<sycl::access::mode::read>(h);
                auto acc_x_next = buf_x_next.get_access<sycl::access::mode::read>(h);
                auto acc_max_diff = buf_max_diff.get_access<sycl::access::mode::write>(h);
                
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    float diff = std::abs(acc_x_next[i] - acc_x_curr[i]);
                    sycl::atomic_ref<float, 
                        sycl::memory_order::relaxed, 
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> atomic_max(acc_max_diff[0]);
                    
                    float old = atomic_max.load();
                    while (diff > old && !atomic_max.compare_exchange_strong(old, diff)) {}
                });
            }).wait();

            float max_diff = 0.0f;
            {
                auto host_acc_max_diff = buf_max_diff.get_host_access();
                max_diff = host_acc_max_diff[0];
            }

            queue.submit([&](sycl::handler& h) {
                auto acc_x_curr = buf_x_curr.get_access<sycl::access::mode::write>(h);
                auto acc_x_next = buf_x_next.get_access<sycl::access::mode::read>(h);
                
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    acc_x_curr[i] = acc_x_next[i];
                });
            }).wait();

            if (max_diff < accuracy) {
                converged = true;
            }
        }

        {
            auto host_acc_x_curr = buf_x_curr.get_host_access();
            for (size_t i = 0; i < n; i++) {
                x_curr[i] = host_acc_x_curr[i];
            }
        }
        
    } catch (sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    }
    
    return x_curr;
}
