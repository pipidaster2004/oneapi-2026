#include "block_gemm_oneapi.h"
#include <cmath>
#include <vector>
#include <iostream>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    std::vector<float> c(size * size, 0.0f);
    
    const size_t BLOCK_SIZE = 16;
    size_t num_blocks = size / BLOCK_SIZE;
    
    try {
        sycl::queue queue(device, sycl::property::queue::in_order{});
        
        float* a_dev = sycl::malloc_device<float>(size * size, queue);
        float* b_dev = sycl::malloc_device<float>(size * size, queue);
        float* c_dev = sycl::malloc_device<float>(size * size, queue);
        
        if (!a_dev || !b_dev || !c_dev) {
            throw std::runtime_error("Failed to allocate device memory");
        }
        
        queue.memcpy(a_dev, a.data(), size * size * sizeof(float));
        queue.memcpy(b_dev, b.data(), size * size * sizeof(float));
        queue.memset(c_dev, 0, size * size * sizeof(float));
        queue.wait();
        
        const size_t wg_size = 16;
        
        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 2> a_block(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::local_accessor<float, 2> b_block(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            
            cgh.parallel_for(sycl::nd_range<2>(
                sycl::range<2>(size, size),
                sycl::range<2>(wg_size, wg_size)),
                [=](sycl::nd_item<2> item) {
                
                size_t i = item.get_global_id(0);
                size_t j = item.get_global_id(1);
                
                if (i >= size || j >= size) return;
                
                size_t bi = i / BLOCK_SIZE;
                size_t bj = j / BLOCK_SIZE;
                size_t li = i % BLOCK_SIZE;
                size_t lj = j % BLOCK_SIZE;
                
                double sum = 0.0;
                
                for (size_t bk = 0; bk < num_blocks; ++bk) {
                    
                    a_block[li][lj] = a_dev[(bi * BLOCK_SIZE + li) * size + (bk * BLOCK_SIZE + lj)];
                    b_block[li][lj] = b_dev[(bk * BLOCK_SIZE + li) * size + (bj * BLOCK_SIZE + lj)];
                    
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                        sum += static_cast<double>(a_block[li][k]) * 
                               static_cast<double>(b_block[k][lj]);
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                c_dev[i * size + j] = static_cast<float>(sum);
            });
        });
        
        queue.wait();
        
        queue.memcpy(c.data(), c_dev, size * size * sizeof(float)).wait();
        
        sycl::free(a_dev, queue);
        sycl::free(b_dev, queue);
        sycl::free(c_dev, queue);
        
    } catch (sycl::exception& e) {
        return std::vector<float>();
    } catch (std::exception& e) {
        return std::vector<float>();
    }
    
    return c;
}