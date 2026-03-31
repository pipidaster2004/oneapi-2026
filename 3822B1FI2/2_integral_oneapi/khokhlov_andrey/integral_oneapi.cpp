#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float result = 0.0f;
  const float step = (end - start) / count;
  sycl::queue compute_queue(device);
  {
    sycl::buffer<float> result_buffer(&result, 1);
    compute_queue.submit([&](sycl::handler& cgh) {
      auto reduction_sum = sycl::reduction(result_buffer, cgh, sycl::plus<>());
      cgh.parallel_for(
        sycl::range<2>(count, count),
        reduction_sum,
        [=](sycl::id<2> index, auto& accumulator) {
          float x_coord = start + step * (index[0] + 0.5f);
          float y_coord = start + step * (index[1] + 0.5f);
          accumulator += sycl::sin(x_coord) * sycl::cos(y_coord);
        }
      );
    }).wait();
  }

  return result * step * step;
}