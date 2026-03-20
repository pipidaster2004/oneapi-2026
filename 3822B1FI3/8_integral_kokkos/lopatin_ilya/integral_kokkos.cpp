#include "integral_kokkos.h"
#include <cmath>
#include <cstdint>

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) {
        return 0.0f;
    }

    using ExecSpace = Kokkos::SYCL;

    const float span = end - start;
    const float step = span / static_cast<float>(count);
    const float half_step = 0.5f * step;
    const std::int64_t cells = static_cast<std::int64_t>(count) * count;

    float accum = 0.0f;

    Kokkos::parallel_reduce(
        "middle_riemann_double_integral",
        Kokkos::RangePolicy<ExecSpace>(0, cells),
        KOKKOS_LAMBDA(const std::int64_t index, float& local_sum) {
            const int row = static_cast<int>(index / count);
            const int col = static_cast<int>(index - static_cast<std::int64_t>(row) * count);

            const float x_mid = start + static_cast<float>(row) * step + half_step;
            const float y_mid = start + static_cast<float>(col) * step + half_step;

            const float value = sinf(x_mid) * cosf(y_mid);
            local_sum += value;
        },
        accum
    );

    return accum * step * step;
}