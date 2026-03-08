#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <vector>
#include <iostream>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    
    try {
        sycl::queue queue(device);
        
        float* a_dev = sycl::malloc_device<float>(n * n, queue);
        float* b_dev = sycl::malloc_device<float>(n, queue);
        float* x_old_dev = sycl::malloc_device<float>(n, queue);
        float* x_new_dev = sycl::malloc_device<float>(n, queue);
        
        if (!a_dev || !b_dev || !x_old_dev || !x_new_dev) {
            throw std::runtime_error("Failed to allocate device memory");
        }
        
        queue.memcpy(a_dev, a.data(), n * n * sizeof(float)).wait();
        queue.memcpy(b_dev, b.data(), n * sizeof(float)).wait();
        
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                x_old_dev[idx] = 0.0f;
            });
        }).wait();
        
        int iteration = 0;
        float max_diff = 0.0f;
        bool converged = false;
        
        std::vector<float> diff_host(n, 0.0f);
        float* diff_dev = sycl::malloc_device<float>(n, queue);
        
        do {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    float sum = 0.0f;
                    
                    for (size_t j = 0; j < n; ++j) {
                        if (j != i) {
                            sum += a_dev[i * n + j] * x_old_dev[j];
                        }
                    }
                    
                    x_new_dev[i] = (b_dev[i] - sum) / a_dev[i * n + i];
                });
            });
            
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    diff_dev[i] = sycl::fabs(x_new_dev[i] - x_old_dev[i]);
                });
            });
            
            queue.memcpy(diff_host.data(), diff_dev, n * sizeof(float)).wait();
            
            max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                if (diff_host[i] > max_diff) {
                    max_diff = diff_host[i];
                }
            }
            
            std::swap(x_old_dev, x_new_dev);
            
            iteration++;
            
        } while (iteration < ITERATIONS && max_diff >= accuracy);
        
        queue.memcpy(x.data(), x_old_dev, n * sizeof(float)).wait();
        
        sycl::free(a_dev, queue);
        sycl::free(b_dev, queue);
        sycl::free(x_old_dev, queue);
        sycl::free(x_new_dev, queue);
        sycl::free(diff_dev, queue);
        
    } catch (sycl::exception& e) {
        return std::vector<float>();
    } catch (std::exception& e) {
        return std::vector<float>();
    }
    
    return x;
}