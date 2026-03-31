#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device dev) {
    const float h = (end - start) / static_cast<float>(count);
    float total_sum = 0.0f;

    sycl::queue compute_queue(dev);

    {
        sycl::buffer<float, 1> result_buffer(&total_sum, 1);

        compute_queue.submit([&](sycl::handler& h_cmd) {
            auto aggregate = sycl::reduction(result_buffer, h_cmd, sycl::plus<float>());

            h_cmd.parallel_for(
                sycl::range<2>(count, count), 
                aggregate, 
                [=](sycl::id<2> index, auto& sum_ref) {
                    float x_mid = start + (static_cast<float>(index[0]) + 0.5f) * h;
                    float y_mid = start + (static_cast<float>(index[1]) + 0.5f) * h;

                    float val = sycl::sin(x_mid) * sycl::cos(y_mid);
                    
                    sum_ref.combine(val);
                }
            );
        });
        compute_queue.wait(); 
    }

    return total_sum * (h * h);
}