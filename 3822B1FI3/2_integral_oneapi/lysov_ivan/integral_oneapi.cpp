#include "integral_oneapi.h"

#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0 || start == end) {
        return 0.0f;
    }

    sycl::queue q(device);

    const float step = (end - start) / static_cast<float>(count);
    const float cell_area = step * step;

    float result = 0.0f;

    {
        sycl::buffer<float, 1> result_buffer(&result, sycl::range<1>(1));

        q.submit([&](sycl::handler& h) {
            auto sum_reduction =
                sycl::reduction(result_buffer, h, sycl::plus<float>());

            h.parallel_for(
                sycl::range<2>(count, count),
                sum_reduction,
                [=](sycl::id<2> idx, auto& sum) {
                    const int j = static_cast<int>(idx[0]);
                    const int i = static_cast<int>(idx[1]);

                    const float x = start + (static_cast<float>(i) + 0.5f) * step;
                    const float y = start + (static_cast<float>(j) + 0.5f) * step;

                    const float value = sycl::sin(x) * sycl::cos(y) * cell_area;
                    sum.combine(value);
                });
        }).wait();
    }

    return result;
}