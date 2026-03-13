#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy) {

    const int n = b.size();

    Kokkos::View<float*> x("x", n);
    Kokkos::View<float*> x_new("x_new", n);
    Kokkos::View<const float*> vec_b(b.data(), n);
    Kokkos::View<const float*> mat_a(a.data(), n * n);

    Kokkos::deep_copy(x, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {

        Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i) {

            float sigma = 0.0f;
            int row = i * n;

            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sigma += mat_a(row + j) * x(j);
                }
            }

            x_new(i) = (vec_b(i) - sigma) / mat_a(row + i);
        }
        );

        float max_diff = 0.0f;

        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i, float& local_max) {
            float diff = fabsf(x_new(i) - x(i));
            if (diff > local_max) local_max = diff;
        },
            Kokkos::Max<float>(max_diff)
        );

        if (max_diff < accuracy) break;

        Kokkos::deep_copy(x, x_new);
    }

    std::vector<float> result(n);

    auto host_view = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(host_view, x);

    for (int i = 0; i < n; ++i)
        result[i] = host_view(i);

    return result;
}