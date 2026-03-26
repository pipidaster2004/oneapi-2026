#include "block_gemm_oneapi.h"
#include <cassert>

std::vector<float> GemmBlockONEAPI(const std::vector<float> &matrix_a,
                                   const std::vector<float> &matrix_b,
                                   size_t matrix_size, sycl::device device) {

  constexpr int kBlockSize = 16;
  assert(matrix_size % kBlockSize == 0);

  sycl::queue computation_queue(device);
  std::vector<float> result_matrix(matrix_size * matrix_size);

  {
    sycl::buffer<float> buffer_a(matrix_a.data(), matrix_a.size());
    sycl::buffer<float> buffer_b(matrix_b.data(), matrix_b.size());
    sycl::buffer<float> buffer_c(result_matrix.data(), result_matrix.size());

    sycl::event compute_event =
        computation_queue.submit([&](sycl::handler &command_handler) {
          sycl::local_accessor<float, 2> block_a(
              sycl::range<2>(kBlockSize, kBlockSize), command_handler);
          sycl::local_accessor<float, 2> block_b(
              sycl::range<2>(kBlockSize, kBlockSize), command_handler);

          auto accessor_a =
              buffer_a.get_access<sycl::access::mode::read>(command_handler);
          auto accessor_b =
              buffer_b.get_access<sycl::access::mode::read>(command_handler);
          auto accessor_c =
              buffer_c.get_access<sycl::access::mode::write>(command_handler);

          command_handler.parallel_for(
              sycl::nd_range<2>(sycl::range<2>(matrix_size, matrix_size),
                                sycl::range<2>(kBlockSize, kBlockSize)),
              [=](sycl::nd_item<2> work_item) {
                int local_row = work_item.get_local_id(0);
                int local_col = work_item.get_local_id(1);
                int global_row = work_item.get_global_id(0);
                int global_col = work_item.get_global_id(1);

                float partial_sum = 0.0f;
                int num_blocks = matrix_size / kBlockSize;

                for (int block_index = 0; block_index < num_blocks;
                     ++block_index) {
                  block_a[local_row][local_col] =
                      accessor_a[global_row * matrix_size +
                                 (kBlockSize * block_index + local_col)];

                  block_b[local_row][local_col] =
                      accessor_b[(kBlockSize * block_index + local_row) *
                                     matrix_size +
                                 global_col];

                  work_item.barrier(sycl::access::fence_space::local_space);

                  for (int k = 0; k < kBlockSize; ++k) {
                    partial_sum +=
                        block_a[local_row][k] * block_b[k][local_col];
                  }

                  work_item.barrier(sycl::access::fence_space::local_space);
                }

                accessor_c[global_row * matrix_size + global_col] = partial_sum;
              });
        });

    compute_event.wait();
  }

  return result_matrix;
}