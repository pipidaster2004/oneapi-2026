#include <cmath>
#include "integral_oneapi.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float result = 0.0f;
    const float step = (end - start) / count;

    sycl::queue queue(device);

    {
        sycl::buffer<float> resultBuffer(&result, 1);

        queue.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(resultBuffer, cgh, sycl::plus<>());

            cgh.parallel_for(
                sycl::range<2>(count, count),
                reduction,
                [=](sycl::id<2> idx, auto& sum) {
                    float x = start + step * (idx.get(0) + 0.5f);
                    float y = start + step * (idx.get(1) + 0.5f);
                    sum += sycl::sin(x) * sycl::cos(y);
                }
            );
        }).wait();
    }

    return result * step * step;
}
