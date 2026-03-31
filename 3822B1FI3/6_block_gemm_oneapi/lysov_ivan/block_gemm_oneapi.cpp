#include "block_gemm_oneapi.h"

#include <algorithm>
#include <vector>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    if (size == 0) {
        return {};
    }

    if (a.size() != size * size || b.size() != size * size) {
        return {};
    }

    sycl::queue q(device);

    std::vector<float> c(size * size, 0.0f);

    constexpr size_t BLOCK_SIZE = 16;

    const size_t global_rows =
        ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    const size_t global_cols =
        ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    sycl::buffer<float, 1> a_buffer(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> b_buffer(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> c_buffer(c.data(), sycl::range<1>(c.size()));

    q.submit([&](sycl::handler& h) {
        auto a_acc = a_buffer.get_access<sycl::access::mode::read>(h);
        auto b_acc = b_buffer.get_access<sycl::access::mode::read>(h);
        auto c_acc = c_buffer.get_access<sycl::access::mode::write>(h);

        sycl::local_accessor<float, 2> tile_a(
            sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
        sycl::local_accessor<float, 2> tile_b(
            sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(global_rows, global_cols),
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)),
            [=](sycl::nd_item<2> item) {
                const size_t row = item.get_global_id(0);
                const size_t col = item.get_global_id(1);

                const size_t local_row = item.get_local_id(0);
                const size_t local_col = item.get_local_id(1);

                float sum = 0.0f;

                for (size_t k0 = 0; k0 < size; k0 += BLOCK_SIZE) {
                    const size_t a_col = k0 + local_col;
                    const size_t b_row = k0 + local_row;

                    if (row < size && a_col < size) {
                        tile_a[local_row][local_col] = a_acc[row * size + a_col];
                    } else {
                        tile_a[local_row][local_col] = 0.0f;
                    }

                    if (b_row < size && col < size) {
                        tile_b[local_row][local_col] = b_acc[b_row * size + col];
                    } else {
                        tile_b[local_row][local_col] = 0.0f;
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                        sum += tile_a[local_row][k] * tile_b[k][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < size && col < size) {
                    c_acc[row * size + col] = sum;
                }
            });
    }).wait();

    return c;
}