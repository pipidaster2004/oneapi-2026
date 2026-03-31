#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    std::vector<float> c(size * size, 0.0f);
    sycl::queue q(device);

    {
        sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> buf_c(c.data(), sycl::range<1>(c.size()));

        float alpha = 1.0f;
        float beta = 0.0f;

        std::int64_t m = size;
        std::int64_t n = size;
        std::int64_t k = size;
        std::int64_t lda = size;
        std::int64_t ldb = size;
        std::int64_t ldc = size;

        oneapi::mkl::blas::column_major::gemm(
            q, 
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            m, n, k, 
            alpha, 
            buf_a, lda, 
            buf_b, ldb, 
            beta, 
            buf_c, ldc
        );

    }

    return c;
}