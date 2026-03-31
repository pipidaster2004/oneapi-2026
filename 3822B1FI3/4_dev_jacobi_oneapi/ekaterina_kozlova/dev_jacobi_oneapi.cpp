#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>


std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& matrix_a, 
        const std::vector<float>& vector_b,
        float precision, 
        sycl::device target_device) {
    
    const int system_size = vector_b.size();

    sycl::queue computation_queue(target_device);
    auto device_context = computation_queue.get_context();
    auto current_device = computation_queue.get_device();

    std::vector<float> solution(system_size, 0.0f);
    std::vector<float> previous_solution(system_size, 0.0f);

    float* device_matrix = sycl::malloc_device<float>(system_size * system_size, computation_queue);
    float* device_rhs = sycl::malloc_device<float>(system_size, computation_queue);
    float* device_current = sycl::malloc_device<float>(system_size, computation_queue);
    float* device_next = sycl::malloc_device<float>(system_size, computation_queue);

    computation_queue.memcpy(device_matrix, matrix_a.data(), 
                             sizeof(float) * system_size * system_size).wait();
    computation_queue.memcpy(device_rhs, vector_b.data(), 
                             sizeof(float) * system_size).wait();
    computation_queue.memset(device_current, 0, sizeof(float) * system_size).wait();
    computation_queue.memset(device_next, 0, sizeof(float) * system_size).wait();

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {

        computation_queue.submit([&](sycl::handler& compute_handler) {
            compute_handler.parallel_for(sycl::range<1>(system_size), 
                [=](sycl::id<1> position) {
                    int row_index = position[0];
                    float accumulated = 0.0f;
                    float diagonal_element = device_matrix[row_index * system_size + row_index];

                    for (int column = 0; column < system_size; ++column) {
                        if (column != row_index) {
                            accumulated += device_matrix[row_index * system_size + column] * 
                                          device_current[column];
                        }
                    }
                    
                    device_next[row_index] = (device_rhs[row_index] - accumulated) / diagonal_element;
                });
        }).wait();

        computation_queue.memcpy(solution.data(), device_next, 
                                sizeof(float) * system_size).wait();
        computation_queue.memcpy(previous_solution.data(), device_current, 
                                sizeof(float) * system_size).wait();

        bool solution_converged = true;
        for (int element = 0; element < system_size; ++element) {
            float difference = std::abs(solution[element] - previous_solution[element]);
            if (difference >= precision) {
                solution_converged = false;
                break;
            }
        }

        computation_queue.memcpy(device_current, solution.data(), 
                                sizeof(float) * system_size).wait();

        if (solution_converged) {
            break;
        }
    }

    sycl::free(device_matrix, computation_queue);
    sycl::free(device_rhs, computation_queue);
    sycl::free(device_current, computation_queue);
    sycl::free(device_next, computation_queue);
    
    return solution;
}
