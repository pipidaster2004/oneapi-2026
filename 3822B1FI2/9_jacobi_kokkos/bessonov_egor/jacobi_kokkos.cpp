#include "jacobi_kokkos.h"

#include <cmath>

std::vector<float> JacobiKokkos(
  const std::vector<float>& a,
  const std::vector<float>& b,
  float accuracy) {

  using ExecSpace = Kokkos::SYCL;
  using MemSpace = Kokkos::SYCLDeviceUSMSpace;

  const int n = b.size();

  Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> a_dev("a_dev", n, n);
  Kokkos::View<float*, MemSpace> b_dev("b_dev", n);
  Kokkos::View<float*, MemSpace> prev_dev("prev_dev", n);
  Kokkos::View<float*, MemSpace> curr_dev("curr_dev", n);

  auto a_host = Kokkos::create_mirror_view(a_dev);
  auto b_host = Kokkos::create_mirror_view(b_dev);

  for (int i = 0; i < n; ++i) {
    b_host(i) = b[i];
    for (int j = 0; j < n; ++j) {
      a_host(i, j) = a[i * n + j];
    }
  }

  Kokkos::deep_copy(a_dev, a_host);
  Kokkos::deep_copy(b_dev, b_host);
  Kokkos::deep_copy(prev_dev, 0.0f);
  Kokkos::deep_copy(curr_dev, 0.0f);

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(int i) {
      float value = b_dev(i);

      for (int j = 0; j < n; ++j) {
        if (i != j) {
          value -= a_dev(i, j) * prev_dev(j);
        }
      }

      curr_dev(i) = value / a_dev(i, i);
    });

    float error = 0.0f;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(int i, float& local_max) {
      float diff = Kokkos::fabs(curr_dev(i) - prev_dev(i));
      if (diff > local_max) {
        local_max = diff;
      }
    },
      Kokkos::Max<float>(error));

    if (error < accuracy) {
      Kokkos::deep_copy(prev_dev, curr_dev);
      break;
    }

    Kokkos::deep_copy(prev_dev, curr_dev);
  }

  auto result_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prev_dev);

  std::vector<float> result(n);
  for (int i = 0; i < n; ++i) {
    result[i] = result_host(i);
  }

  return result;
}