#ifndef __JACOBI_KOKKOS_H
#define __JACOBI_KOKKOS_H

#include <vector>

#include <Kokkos_Core.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiKokkos(
        const std::vector<float> a,
        const std::vector<float> b,
        float accuracy);

#endif  // __JACOBI_KOKKOS_H
