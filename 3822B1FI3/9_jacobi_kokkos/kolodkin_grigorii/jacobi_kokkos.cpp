#include "jacobi_kokkos.h"
#include <cmath>
#include <algorithm>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {

    size_t total_size = a.size();
    size_t n = static_cast<size_t>(std::sqrt(total_size));

    Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space> A("A", n, n);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> B("B", n);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> X("X", n);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> X_new("X_new", n);
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> X_old("X_old", n);

    auto A_host = Kokkos::create_mirror_view(A);
    auto B_host = Kokkos::create_mirror_view(B);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_host(i, j) = a[i * n + j];
        }
        B_host(i) = b[i];
    }
    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(B, B_host);

    Kokkos::deep_copy(X, 0.0f);
    Kokkos::deep_copy(X_new, 0.0f);

    auto jacobi_kernel = KOKKOS_LAMBDA(const int i) {
        float sum = 0.0f;
        float a_ii = A(i, i);
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                sum += A(i, j) * X(j);
            }
        }
        X_new(i) = (B(i) - sum) / a_ii;
    };

    bool converged = false;
    for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {
        Kokkos::deep_copy(X_old, X);

        Kokkos::parallel_for("Jacobi Iteration", n, jacobi_kernel);
        Kokkos::fence();

        Kokkos::deep_copy(X, X_new);

        auto X_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X);
        auto X_old_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X_old);

        converged = true;
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(X_host(i) - X_old_host(i)) >= accuracy) {
                converged = false;
                break;
            }
        }
    }

    std::vector<float> result(n);
    auto X_final = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, X);
    for (size_t i = 0; i < n; ++i) {
        result[i] = X_final(i);
    }

    Kokkos::finalize();

    return result;
}

