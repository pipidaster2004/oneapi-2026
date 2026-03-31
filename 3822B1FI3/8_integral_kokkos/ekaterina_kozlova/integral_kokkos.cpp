#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    using ExecSpace = Kokkos::SYCL;
    
    const float step_size = (end - start) / static_cast<float>(count);
    const float rect_area = step_size * step_size;

    double sin_accum = 0.0;
    Kokkos::parallel_reduce(
        "SinIntegral",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int idx, double& local_sum) {
            const float point = start + step_size * (static_cast<float>(idx) + 0.5f);
            local_sum += Kokkos::sin(point);
        },
        sin_accum
    );

    double cos_accum = 0.0;
    Kokkos::parallel_reduce(
        "CosIntegral",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int idx, double& local_sum) {
            const float point = start + step_size * (static_cast<float>(idx) + 0.5f);
            local_sum += Kokkos::cos(point);
        },
        cos_accum
    );

    return static_cast<float>(sin_accum * cos_accum * rect_area);
}