#include "block_gemm_oneapi.h"
#include <algorithm>

#define TILE_DIM 16
#define PADDED_TILE (TILE_DIM + 1)

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    sycl::queue compute_queue(device, sycl::property::queue::in_order{});

    float* dev_a = sycl::aligned_alloc_device<float>(64, size * size, compute_queue);
    float* dev_b = sycl::aligned_alloc_device<float>(64, size * size, compute_queue);
    float* dev_c = sycl::aligned_alloc_device<float>(64, size * size, compute_queue);

    compute_queue.memcpy(dev_a, a.data(), size * size * sizeof(float));
    compute_queue.memcpy(dev_b, b.data(), size * size * sizeof(float));
    compute_queue.memset(dev_c, 0, size * size * sizeof(float));
    compute_queue.wait();

    const size_t num_tiles = size / TILE_DIM;

    compute_queue.submit([&](sycl::handler& cmd_group) {
        sycl::local_accessor<float, 2> tile_a(
            sycl::range<2>(TILE_DIM, PADDED_TILE), cmd_group);
        sycl::local_accessor<float, 2> tile_b(
            sycl::range<2>(PADDED_TILE, TILE_DIM), cmd_group);

        cmd_group.parallel_for(sycl::nd_range<2>(
            sycl::range<2>(num_tiles * TILE_DIM, num_tiles * TILE_DIM),
            sycl::range<2>(TILE_DIM, TILE_DIM)
        ), [=](sycl::nd_item<2> thread_idx) {
            const size_t tile_row = thread_idx.get_group(0);
            const size_t tile_col = thread_idx.get_group(1);
            const size_t thread_row = thread_idx.get_local_id(0);
            const size_t thread_col = thread_idx.get_local_id(1);

            float partial_sum = 0.0f;

            for (size_t tile_k = 0; tile_k < num_tiles; tile_k++) {
                tile_a[thread_row][thread_col] = 
                    dev_a[(tile_row * TILE_DIM + thread_row) * size + 
                          tile_k * TILE_DIM + thread_col];
                tile_b[thread_row][thread_col] = 
                    dev_b[(tile_k * TILE_DIM + thread_row) * size + 
                          tile_col * TILE_DIM + thread_col];

                thread_idx.barrier(sycl::access::fence_space::local_space);

                #pragma unroll
                for (size_t elem_idx = 0; elem_idx < TILE_DIM; elem_idx++) {
                    partial_sum += tile_a[thread_row][elem_idx] * 
                                   tile_b[elem_idx][thread_col];
                }

                thread_idx.barrier(sycl::access::fence_space::local_space);
            }

            const size_t global_row = tile_row * TILE_DIM + thread_row;
            const size_t global_col = tile_col * TILE_DIM + thread_col;
            dev_c[global_row * size + global_col] = partial_sum;
        });
    }).wait();

    std::vector<float> output(size * size);
    compute_queue.memcpy(output.data(), dev_c, size * size * sizeof(float)).wait();

    sycl::free(dev_a, compute_queue);
    sycl::free(dev_b, compute_queue);
    sycl::free(dev_c, compute_queue);

    return output;
}
