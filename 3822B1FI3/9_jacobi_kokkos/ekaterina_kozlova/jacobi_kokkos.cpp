#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    
    using ExecType = Kokkos::SYCL;
    using MemType = Kokkos::SYCLDeviceUSMSpace;

    const int dim = static_cast<int>(b.size());
    const float acc_sq = accuracy * accuracy;

    Kokkos::View<float**, Kokkos::LayoutLeft, MemType> mat_A("MatrixA", dim, dim);
    Kokkos::View<float*, MemType> vec_B("VectorB", dim);
    Kokkos::View<float*, MemType> inv_diag("InvDiagonal", dim);
    Kokkos::View<float*, MemType> sol_old("SolutionOld", dim);
    Kokkos::View<float*, MemType> sol_new("SolutionNew", dim);

    auto host_mat = Kokkos::create_mirror_view(mat_A);
    auto host_vec = Kokkos::create_mirror_view(vec_B);
    
    for (int idx = 0; idx < dim; idx++) {
        host_vec(idx) = b[idx];
        for (int jdx = 0; jdx < dim; jdx++) {
            host_mat(idx, jdx) = a[idx * dim + jdx];
        }
    }
    
    Kokkos::deep_copy(mat_A, host_mat);
    Kokkos::deep_copy(vec_B, host_vec);

    Kokkos::parallel_for("InitDiagValues", Kokkos::RangePolicy<ExecType>(0, dim),
        KOKKOS_LAMBDA(int pos) {
            inv_diag(pos) = 1.0f / mat_A(pos, pos);
            sol_old(pos) = 0.0f;
        });

    bool done = false;
    const int freq = 8;

    for (int step = 0; step < ITERATIONS && !done; step++) {
        Kokkos::parallel_for("JacobiStep", Kokkos::RangePolicy<ExecType>(0, dim),
            KOKKOS_LAMBDA(int eq) {
                float total = 0.0f;
                for (int var = 0; var < dim; var++) {
                    if (var != eq) {
                        total += mat_A(eq, var) * sol_old(var);
                    }
                }
                sol_new(eq) = inv_diag(eq) * (vec_B(eq) - total);
            });

        if ((step + 1) % freq == 0) {
            float max_err = 0.0f;
            Kokkos::parallel_reduce("CheckConvergence",
                Kokkos::RangePolicy<ExecType>(0, dim),
                KOKKOS_LAMBDA(int loc, float& thread_val) {
                    float delta = Kokkos::fabs(sol_new(loc) - sol_old(loc));
                    if (delta > thread_val) thread_val = delta;
                },
                Kokkos::Max<float>(max_err)
            );

            if (max_err < accuracy) {
                done = true;
                break;
            }
        }

        Kokkos::kokkos_swap(sol_old, sol_new);
    }

    std::vector<float> res(dim);
    auto host_res = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol_old);
    for (int comp = 0; comp < dim; comp++) {
        res[comp] = host_res(comp);
    }

    return res;
}