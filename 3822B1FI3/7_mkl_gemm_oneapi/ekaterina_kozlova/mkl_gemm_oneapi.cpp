#include "mkl_gemm_oneapi.h"
#include <cstdint>
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    sycl::queue q(device, sycl::property::queue::in_order{});

    const size_t total = size * size;
    const std::int64_t n = static_cast<std::int64_t>(size);
    const std::int64_t lda = static_cast<std::int64_t>(size);
    const std::int64_t ldb = static_cast<std::int64_t>(size);
    const std::int64_t ldc = static_cast<std::int64_t>(size);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::vector<float> result(total);

    float* dev_a = sycl::aligned_alloc_device<float>(64, total, q);
    float* dev_b = sycl::aligned_alloc_device<float>(64, total, q);
    float* dev_c = sycl::aligned_alloc_device<float>(64, total, q);

    q.memcpy(dev_a, a.data(), total * sizeof(float));
    q.memcpy(dev_b, b.data(), total * sizeof(float));

    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, alpha, dev_a, lda, dev_b, ldb, beta, dev_c, ldc);

    q.memcpy(result.data(), dev_c, total * sizeof(float)).wait();

    sycl::free(dev_a, q);
    sycl::free(dev_b, q);
    sycl::free(dev_c, q);

    return result;
}
