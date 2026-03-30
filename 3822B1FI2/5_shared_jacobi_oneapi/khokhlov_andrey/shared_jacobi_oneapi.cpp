#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy,
        sycl::device device) {
    
    const size_t matrix_size = b.size();

    if (matrix_size == 0 || a.size() != matrix_size * matrix_size) {
        return {};
    }

    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }
    
    const float squared_accuracy = accuracy * accuracy;

    sycl::queue compute_queue(device, sycl::property::queue::in_order{});

    float* shared_matrix = sycl::malloc_shared<float>(matrix_size * matrix_size, compute_queue);
    float* shared_rhs = sycl::malloc_shared<float>(matrix_size, compute_queue);
    float* shared_inv_diag = sycl::malloc_shared<float>(matrix_size, compute_queue);
    float* shared_current_x = sycl::malloc_shared<float>(matrix_size, compute_queue);
    float* shared_next_x = sycl::malloc_shared<float>(matrix_size, compute_queue);
    float* shared_max_diff = sycl::malloc_shared<float>(1, compute_queue);

    if (!shared_matrix || !shared_rhs || !shared_inv_diag || 
        !shared_current_x || !shared_next_x || !shared_max_diff) {
        sycl::free(shared_matrix, compute_queue);
        sycl::free(shared_rhs, compute_queue);
        sycl::free(shared_inv_diag, compute_queue);
        sycl::free(shared_current_x, compute_queue);
        sycl::free(shared_next_x, compute_queue);
        sycl::free(shared_max_diff, compute_queue);
        return {};
    }

    for (size_t i = 0; i < a.size(); ++i) {
        shared_matrix[i] = a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        shared_rhs[i] = b[i];
    }

    for (size_t i = 0; i < matrix_size; ++i) {
        float diagonal = shared_matrix[i * matrix_size + i];
        shared_inv_diag[i] = (std::fabs(diagonal) > 1e-12f) ? (1.0f / diagonal) : 0.0f;
        shared_current_x[i] = 0.0f;
        shared_next_x[i] = 0.0f;
    }

    compute_queue.wait();

    const size_t work_group_size = 128;
    const size_t global_size = ((matrix_size + work_group_size - 1) / work_group_size) * work_group_size;
    
    bool is_converged = false;
    int check_frequency = 4;

    for (int iteration = 0; iteration < ITERATIONS && !is_converged; ++iteration) {
        *shared_max_diff = 0.0f;
        compute_queue.parallel_for(
            sycl::nd_range<1>(global_size, work_group_size),
            sycl::reduction(shared_max_diff, sycl::maximum<float>()),
            [=](sycl::nd_item<1> work_item, auto& max_reduction) {
                size_t row_index = work_item.get_global_id(0);
                if (row_index >= matrix_size) return;
                float sum_all = 0.0f;
                const size_t row_start = row_index * matrix_size;
                #pragma unroll 4
                for (size_t col_index = 0; col_index < matrix_size; ++col_index) {
                    sum_all += shared_matrix[row_start + col_index] * shared_current_x[col_index];
                }
                float diagonal = shared_matrix[row_start + row_index];
                float sum_off_diagonal = sum_all - diagonal * shared_current_x[row_index];
                float new_value = shared_inv_diag[row_index] * (shared_rhs[row_index] - sum_off_diagonal);
                shared_next_x[row_index] = new_value;
                float difference = sycl::fabs(new_value - shared_current_x[row_index]);
                max_reduction.combine(difference);
            }
        );

        compute_queue.wait();

        if (*shared_max_diff < accuracy) {
            is_converged = true;
            break;
        }

        std::swap(shared_current_x, shared_next_x);

        if ((iteration + 1) % check_frequency == 0 && iteration > 0) {
            float norm_difference = 0.0f;
            for (size_t i = 0; i < matrix_size; ++i) {
                float diff = shared_current_x[i] - shared_next_x[i];
                norm_difference += diff * diff;
            }
            
            if (norm_difference < squared_accuracy) {
                is_converged = true;
                break;
            }
        }
    }

    std::vector<float> result(matrix_size);
    for (size_t i = 0; i < matrix_size; ++i) {
        result[i] = shared_current_x[i];
    }

    sycl::free(shared_matrix, compute_queue);
    sycl::free(shared_rhs, compute_queue);
    sycl::free(shared_inv_diag, compute_queue);
    sycl::free(shared_current_x, compute_queue);
    sycl::free(shared_next_x, compute_queue);
    sycl::free(shared_max_diff, compute_queue);
    
    return result;
}