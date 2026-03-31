#include "acc_jacobi_oneapi.h"
#include <algorithm>

using buftype = sycl::buffer<float>;

std::vector<float> JacobiAccONEAPI(const std::vector<float> &a,
                                   const std::vector<float> &b, float accuracy,
                                   sycl::device device) {
  size_t size = b.size();
  std::vector<float> prev_res(size, 0.0f);
  std::vector<float> res(size, 0.0f);

  auto stop = [&]() -> bool {
    float norm = 0.0f;
    for (int idx = 0; idx < prev_res.size(); ++idx) {
      float el = res[idx] - prev_res[idx];
      norm += el * el;
    }
    return norm < accuracy * accuracy;
  };

  sycl::queue gpu_queue(device);

  for (int epoch = 0; epoch < ITERATIONS; ++epoch) {

    buftype buf_a(a.data(), a.size());
    buftype buf_b(b.data(), b.size());
    buftype buf_prev_res(prev_res.data(), prev_res.size());
    buftype buf_res(res.data(), res.size());

    gpu_queue
        .submit([&](sycl::handler &cgh) {
          auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
          auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
          auto in_prev_res =
              buf_prev_res.get_access<sycl::access::mode::read>(cgh);
          auto in_res = buf_res.get_access<sycl::access::mode::write>(cgh);

          cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
            int idx = id.get(0);
            float next_res = 0;

            for (int indx = 0; indx < size; indx++) {
              if (idx == indx) {
                next_res += in_b[indx];
              } else {
                next_res -= in_a[idx * size + indx] * in_prev_res[indx];
              }
            }
            next_res /= in_a[idx * size + idx];
            in_res[idx] = next_res;
          });
        })
        .wait();

    std::swap(prev_res, res);
    if (stop()) {
      break;
    }

  }
  return prev_res;
}
