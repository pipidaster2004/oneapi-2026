#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    std::vector<float> c(size * size, 0.0f);
    sycl::queue q(device);

    const size_t tile_size = 16;

    {
        sycl::buffer<float, 2> buf_a(a.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_b(b.data(), sycl::range<2>(size, size));
        sycl::buffer<float, 2> buf_c(c.data(), sycl::range<2>(size, size));

        q.submit([&](sycl::handler& h) {
            auto A = buf_a.get_access<sycl::access::mode::read>(h);
            auto B = buf_b.get_access<sycl::access::mode::read>(h);
            auto C = buf_c.get_access<sycl::access::mode::write>(h);

            sycl::local_accessor<float, 2> tile_a(sycl::range<2>(tile_size, tile_size), h);
            sycl::local_accessor<float, 2> tile_b(sycl::range<2>(tile_size, tile_size), h);

            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(size, size), 
                                            sycl::range<2>(tile_size, tile_size)), 
                [=](sycl::nd_item<2> item) {
                
                size_t row = item.get_global_id(0);
                size_t col = item.get_global_id(1);
                size_t local_row = item.get_local_id(0);
                size_t local_col = item.get_local_id(1);

                float sum = 0.0f;

                for (size_t k = 0; k < size; k += tile_size) {
                    tile_a[local_row][local_col] = A[row][k + local_col];
                    tile_b[local_row][local_col] = B[k + local_row][col];

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t i = 0; i < tile_size; ++i) {
                        sum += tile_a[local_row][i] * tile_b[i][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                C[row][col] = sum;
            });
        });
    }

    return c;
}