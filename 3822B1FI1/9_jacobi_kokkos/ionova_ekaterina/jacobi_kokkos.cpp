#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {

    const size_t dim = b.size();

    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> mat("matrix", dim, dim);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> rhs("rhs", dim);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> current("current", dim);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> next("next", dim);

    auto mat_h = Kokkos::create_mirror_view(mat);
    auto rhs_h = Kokkos::create_mirror_view(rhs);
    for (size_t r = 0; r < dim; ++r) {
        rhs_h(r) = b[r];
        for (size_t c = 0; c < dim; ++c) mat_h(r, c) = a[r * dim + c];
    }
    Kokkos::deep_copy(mat, mat_h);
    Kokkos::deep_copy(rhs, rhs_h);
    Kokkos::deep_copy(current, 0.0f);

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        float max_diff = 0.0f;

        Kokkos::parallel_reduce("JacobiStep", dim, KOKKOS_LAMBDA(const int row, float& local_max) {
            float accum = 0.0f;
            for (int col = 0; col < (int)dim; ++col) {
                if (col != row) accum += mat(row, col) * current(col);
            }

            float res = (rhs(row) - accum) / mat(row, row);
            float diff = Kokkos::abs(res - current(row));
            
            next(row) = res;
            if (diff > local_max) local_max = diff;
        }, Kokkos::Max<float>(max_diff));

        std::swap(current, next);

        if (max_diff < accuracy) break;
    }

    std::vector<float> solution(dim);
    auto final_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, current);
    for (size_t i = 0; i < dim; ++i) solution[i] = final_host(i);

    return solution;
}