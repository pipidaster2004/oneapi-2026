#include "shared_jacobi_oneapi.h"
#include <algorithm>

using buftype = sycl::buffer<float>;

std::vector<float> JacobiSharedONEAPI(const std::vector<float> &a,
                                      const std::vector<float> &b,
                                      float accuracy, sycl::device device) {
  size_t size = b.size();
  std::vector<float> res(size, 0.0f);

  sycl::queue gpu_queue(device);

  auto alloc = [&gpu_queue](size_t size) -> float * {
    return sycl::malloc_shared<float>(size, gpu_queue);
  };

  auto free = [&gpu_queue](void *ptr) -> void { sycl::free(ptr, gpu_queue); };

  float *in_a = alloc(a.size());
  float *in_b = alloc(b.size());
  float *in_prev_res = alloc(size);
  float *in_res = alloc(res.size());

  auto stop = [&]() -> bool {
    float norm = 0.0f;
    for (int idx = 0; idx < size; ++idx) {
      float el = in_res[idx] - in_prev_res[idx];
      norm += el * el;
    }
    return norm < accuracy * accuracy;
  };

  std::copy(a.data(), a.data() + a.size(), in_a);
  std::copy(b.data(), b.data() + b.size(), in_b);

  std::fill(in_prev_res, in_prev_res + size, 0.0f);
  std::fill(in_res, in_res + size, 0.0f);

  for (int epoch = 0; epoch < ITERATIONS; ++epoch) {

    gpu_queue
        .submit([&](sycl::handler &cgh) {
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

    if (stop()) {
      break;
    }
    std::copy(in_res, in_res + size, in_prev_res);
  }

  std::copy(in_res, in_res + size, res.data());

  free(in_a);
  free(in_b);
  free(in_prev_res);
  free(in_res);

  return res;
}
