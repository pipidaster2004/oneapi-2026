#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>


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

int main()
{
    const size_t N = 4096;          
    const size_t NN = N * N;

    std::vector<float> A(NN);
    for (size_t i = 0; i < N; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            if (i != j) {
                A[i * N + j] = 1.0f;
                row_sum += 1.0f;
            }
        }
        A[i * N + i] = row_sum + 1.0f;
    }

    std::vector<float> b(N);
    for (size_t i = 0; i < N; ++i)
        b[i] = static_cast<float>(i) + 1.0f;
// Получаем доступные устройства
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> devices;
    
    for (auto& platform : platforms) {
        auto platform_devices = platform.get_devices();
        devices.insert(devices.end(), platform_devices.begin(), platform_devices.end());
    }
    
    if (devices.empty()) {
        std::cerr << "No SYCL devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "\nAvailable devices:" << std::endl;
    for (size_t i = 0; i < devices.size(); i++) {
        std::cout << i << ": " << devices[i].get_info<sycl::info::device::name>() << std::endl;
    }
    
    // Используем первое устройство (обычно GPU если доступно)
    sycl::device device = devices[0];
    std::cout << "\nUsing device: " << device.get_info<sycl::info::device::name>() << std::endl;
    float accuracy = 1e-5f;

    auto start = std::chrono::high_resolution_clock::now();

    auto solution = JacobiAccONEAPI(A, b, accuracy, device);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Jacobi method finished in " << elapsed.count() << " seconds\n";

    std::cout << std::fixed;
    std::cout.precision(6);
    std::cout << "Solution sample: ";
    for (int i = 0; i < std::min<size_t>(10, solution.size()); ++i) {
        std::cout << solution[i] << " ";
    }
    std::cout << "\n";

    return 0;
}



