#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {

  float result = 0.0;
  const float scale = (end - start) / count;
  sycl::queue gpu_queue(device);

  {
    sycl::buffer<float> buf(&result, 1);

    gpu_queue.submit([&](sycl::handler &cgh) {
          auto reduction = sycl::reduction(buf, cgh, sycl::plus<>());

          cgh.parallel_for(sycl::range<2>(count, count), reduction,
                          [=](sycl::id<2> id, auto &res) {
                            float val_x = start + scale * (id.get(0) + 0.5f);
                            float val_y = start + scale * (id.get(1) + 0.5f);
                            res += sycl::sin(val_x) * sycl::cos(val_y);
                          });
        })
        .wait();
  }

  return result * scale * scale;
}
