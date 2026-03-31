# Content
- [How To](#how-to)
- [Configuration](#configuration)
- [Time Measurement](#time-measurement)
- [Tasks](#tasks)
- [Results](#results)

# How To
1. Create [github](https://github.com/) account (if not exists);
2. Make sure SSH clone & commit is working ([Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh));
3. Fork this repo (just click **Fork** button on the top of the page, detailed instructions [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project))
4. Clone your forked repo into your local machine, use your user instead of `username`:
```sh
git clone git@github.com:username/oneapi-2026.git
cd oneapi-2026
```
5. Go to your group folder, e.g.:
```sh
cd 3821B1FI1
```
6. Go to needed task folder, e.g.:
```sh
cd 1_integral_oneapi
```
7. Create new folder with your surname and name (**make sure it's the same for all tasks**), e.g.:
```sh
mkdir petrov_ivan
```
8. Copy your task source/header files (including main program) into this folder (use `copy` instead of `cp` on Windows), e.g.:
```sh
cd petrov_ivan
cp /home/usr/lab/*.cpp .
cp /home/usr/lab/*.h .
```
8. Push your sources to github repo, e.g.:
```sh
cd ..
git add .
git commit -m "1_integral_oneapi task"
git push
```
9. Go to your repo in browser, click **Contribute** button on the top of page, then **Open pull request**. Provide meaningfull request title and description, then **Create pull request** (see details [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)).
10. Go to Pull Requests [page](https://github.com/avgorshk/oneapi-2025/pulls) in course repo, find your pull request and check if there are no any merge conflicts occur. If merge conflicts happen - resolve it following the instruction provided by github.

# Time Measurement
The following scheme is used to measure task execution time:
```cpp
int main() {
    // ...

    // Warming-up
    Task(input, size / 8);

    // Performance Measuring
    auto start = std::chrono::high_resolution_clock::now();
    auto c = Task(input, size);
    auto end = std::chrono::high_resolution_clock::now();

    // ...
}
```

# Configuration
- CPU: Intel Core i5 12600K (4 cores, 4 threads)
- RAM: 16 GB
- GPU: Intel UHD Graphics 770 (8 GB)
- Host Compiler: GCC 11.4.0
- oneAPI: 2025.1
- Kokkos: 4.7.1

# Tasks
## Task #1: Permutations
To train modern C++11 skills, the following task is suggested.

There is a set of strings, each string is unique and contains small English letters only. The goal is to make a dictionary of permutations - for each string in a set one should find all other strings from the same set that are permutations of this string.

The following function should be implemented:
```cpp
using dictionary_t = std::map<std::string, std::vector<std::string>>;
void Permutations(dictionary_t& dictionary);
```
Initially, dictionary will contain key strings only (all vectors will be empty). After function completion, the same dictionary should additionally keep the lists of permutations for each key string. Each list of permutations should be sorted in reverse alphabetical order.

The following example will illustrate the idea.
Let's consider the following set of strings as an input:
```
aaa
acb
acd
ad
adc
bac
bc
bcc
bd
bda
bdc
caa
cad
cb
cc
ccb
cd
dac
db
dc
dca
dcb
dcc
dd
```
As a result, one should get the following:
```
aaa :
acb : bac
acd : dca dac cad adc
ad :
adc : dca dac cad acd
bac : acb
bc : cb
bcc : ccb
bd : db
bda :
bdc : dcb
caa :
cad : dca dac adc acd
cb : bc
cc :
ccb : bcc
cd : dc
dac : dca cad adc acd
db : bd
dc : cd
dca : dac cad adc acd
dcb : bdc
dcc :
dd :
```
Two files are expected to be uploaded:
- permutations_cxx.h
```cpp
#ifndef __PERMUTATIONS_CXX_H
#define __PERMUTATIONS_CXX_H

#include <map>
#include <string>
#include <vector>

using dictionary_t = std::map<std::string, std::vector<std::string>>;

void Permutations(dictionary_t& dictionary);

#endif  // __PERMUTATIONS_CXX_H
```
- permutations_cxx.cpp
```cpp
#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
    // Place your implementation here
}
```

## Task #2: Double Integral Computation
In many cases there is no analytical solution for the integral, but one may use approximation of any kind. One of such approximations could be retrieved with the help of [Riemann Sum](https://en.wikipedia.org/wiki/Riemann_sum).

E.g., for double integral the following formula could be used to get its approximate value:

$\int_a^b\int_c^df(x,y)dxdy=\sum_{j=0}^{n-1}\sum_{i=0}^{n-1}f(\frac{x_i+x_{i+1}}2, \frac{y_j+y_{j+1}}2)(x_{i+1}-x_i)(y_{j+1}-y_j)$

The task goal is to compute the following integral using **Middle Riemann Sum**:

$\int_{start}^{end}\int_{start}^{end}sin(x)cos(y)dxdy$

**Hint**: for $start=0$ and $end=1$ it should be equal to $0.3868223$.

Implement the function in SYCL with the following interface:
```cpp
float IntegralONEAPI(float start, float end, int count, sycl::device device);
```
$Count$ means how many intervals one should use to split integration space (the same for $x$ and $y$). E.g. if $count=10$, one will have $10*10=100$ rectangles in total.

Two files are expected to be uploaded:
- integral_oneapi.h
```cpp
#ifndef __INTEGRAL_ONEAPI_H
#define __INTEGRAL_ONEAPI_H

#include <sycl/sycl.hpp>

float IntegralONEAPI(float start, float end, int count, sycl::device device);

#endif  // __INTEGRAL_ONEAPI_H
```
- integral_oneapi.cpp
```cpp
#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    // Place your implementation here
}
```

## Task #3: Jacobi Method (Accessors)
Systems of linear equations are basic apparatus (as part of mathematical model) for variety of problems in physics, chemistry, economics, etc.

There two methods of their solving - direct and iterative. Direct methods (like Gaussian Elimination) are able to get accurate solution, while iterative ones can only provide approximate results.
At the same time, in practice iterative methods may be preferable, e.g. in case of huge matrices that have to stored in RAM while using direct approaches, or if one has close enough initial estimation for final result.

One of such iterative methods is [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method) that allows to get accurate enough solution using the following formula:

$x_i^{(k+1)}=\frac{1}a_{ii}(b_i-\sum_{j \neq i}a_{ij}x_j^{(k)})$

Here $x^{(k+1)}$ is the next approximation of system solution, computed from the previous  $x^{(k)}$. First approximation $x^{(0)}$ could be all zeros.

There are two ways to stop computations:
1. After $N$ iterations, where $N$ is some predefined constant;
2. If $|x^{(k+1)}-x^{(k)}|<Eps$, where $Eps$ is target accuracy.

**Note:** to ensure method convergence one should use it only for [strictly diagonally dominant](https://en.wikipedia.org/wiki/Diagonally_dominant_matrix) system, that means:

$|a_{ii}|>\sum_{j \neq i}|a_{ij}|$ for any $i$.

To complete this task, one should implement the function that computes the solution for the system of linear equations using Jacobi method:
```cpp
std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device);
```
One should implement both stop methods at the same time. Use $N=1024$ as maximum iterations count and $accuracy$ argument as $Eps$. computations have to be stopped when $|x^{(k+1)}-x^{(k)}|<accuracy$ first, and if it's not happening, when after 1024 iterations.

Matix $a$ is stored by rows. One should implement the algorithm using SYCL buffers & accessors approach.

Two files are expected to be uploaded:
- acc_jacobi_oneapi.h
```cpp
#ifndef __ACC_JACOBI_ONEAPI_H
#define __ACC_JACOBI_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device);

#endif  // __ACC_JACOBI_ONEAPI_H
```
- acc_jacobi_oneapi.cpp
```cpp
#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    // Place your implementation here
}
```

## Task #4: Jacobi Method (Device Memory)
This task assumes Jacobi method implementation using SYCL device memory approach (see all the details in Task #2).

Two files are expected to be uploaded:
- dev_jacobi_oneapi.h
```cpp
#ifndef __DEV_JACOBI_ONEAPI_H
#define __DEV_JACOBI_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device);

#endif  // __DEV_JACOBI_ONEAPI_H
```
- dev_jacobi_oneapi.cpp
```cpp
#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    // Place your implementation here
}
```

## Task #5: Jacobi Method (Shared Memory)
This task assumes Jacobi method implementation using SYCL shared memory approach (see all the details in Task #2).

Two files are expected to be uploaded:
- shared_jacobi_oneapi.h
```cpp
#ifndef __SHARED_JACOBI_ONEAPI_H
#define __SHARED_JACOBI_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device);

#endif  // __SHARED_JACOBI_ONEAPI_H
```
- shared_jacobi_oneapi.cpp
```cpp
#include "shared_jacobi_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    // Place your implementation here
}
```

## Task #6: Block Matrix Multiplication
General matrix multiplication (GEMM) is a very basic and broadly used linear algebra operation applied in high performance computing (HPC), statistics, deep learning and other domains. There are a lot of GEMM algorithms with different mathematical complexity form $O(n^3)$ for naive and block approaches to $O(n^{2.371552})$ for the method descibed by Williams et al. in 2024 [[1](https://epubs.siam.org/doi/10.1137/1.9781611977912.134)]. But despite a variety of algorithms with low complexity, block matrix multiplication remains the most used implementation in practice since it fits to modern HW better.

In real applications block-based approach for matrix multiplication can get multiple times faster execution comparing with naive version due to cache friendly approach.

In block version, algorithm could be divided into three stages:
1. Split matricies into blocks (block size normally affects performance significantly so choose it consciously);
2. Multiply two blocks to get partial result;
3. Replay step 2 for all row/column blocks accumulating values into a single result block.

From math perspective, block matrix multiplication could be described by the following formula, where $C_{IJ}$, $A_{IK}$ and $B_{KJ}$ are sub-matricies with the size $block\_size*block\_size$:

$C_{IJ}=\sum_{k=1}^{block_count}A_{IK}B_{KJ}$

Each matrix must be stored in a linear array by rows, so that `a.size()==size*size`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2 and all matricies are square.

Two files are expected to be uploaded:
- block_gemm_oneapi.h:
```cpp
#ifndef __BLOCK_GEMM_ONEAPI_H
#define __BLOCK_GEMM_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device);

#endif  // __BLOCK_GEMM_ONEAPI_H
```
- block_gemm_oneapi.cpp:
```cpp
#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    // Place your implementation here
}
```

## Task #7: Matrix Multiplication Using oneMKL
The most performant way to multiply two matrices on particular hardware is to use vendor-provided library for this purpose. In SYCL it's [oneMKL](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/onemkl/source/). Try to use oneMKL BLAS API to implement general matrix multiplication in most performant way.

Each matrix must be stored in a linear array by rows, so that `a.size()==size*suze`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2 and all matricies are square.

**Note**, that in oneMKL BLAS API matrix is expected to be stored by columns, so additional transpose (or slightly different API) may be required.

Two files are expected to be uploaded:
- mkl_gemm_oneapi.h:
```cpp
#ifndef __MKL_GEMM_ONEAPI_H
#define __MKL_GEMM_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device);

#endif  // __MKL_GEMM_ONEAPI_H
```
- mkl_gemm_oneapi.cpp:
```cpp
#include "mkl_gemm_oneapi.h"

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    // Place your implementation here
}
```
## Task #8: Double Integral Computation In Kokkos
Compute the following integral using **Middle Riemann Sum**:

$\int_{start}^{end}\int_{start}^{end}sin(x)cos(y)dxdy$

**Hint**: for $start=0$ and $end=1$ it should be equal to $0.3868223$.

Implement the function in Kokkos with the following interface:
```cpp
float IntegralKokkos(float start, float end, int count);
```
$Count$ means how many intervals one should use to split integration space (the same for $x$ and $y$). E.g. if $count=10$, one will have $10*10=100$ rectangles in total.

Function should work for SYCL backend. Default execution space is `Kokkos::SYCL`. Default memory space is `Kokkos::SYCLDeviceUSMSpace`. Target device is Intel Graphics.

Two files are expected to be uploaded:
- integral_kokkos.h
```cpp
#ifndef __INTEGRAL_KOKKOS_H
#define __INTEGRAL_KOKKOS_H

#include <Kokkos_Core.hpp>

float IntegralKokkos(float start, float end, int count);

#endif  // __INTEGRAL_KOKKOS_H
```
- integral_kokkos.cpp
```cpp
#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    // Place your implementation here
}
```
## Task #9: Jacobi Method in Kokkos
This task assumes Jacobi method implementation using Kokkos (see all the details in Task #2).

As previously, function should work for SYCL backend. Default execution space is `Kokkos::SYCL`. Default memory space is `Kokkos::SYCLDeviceUSMSpace`. Target device is Intel Graphics.

Two files are expected to be uploaded:
- jacobi_kokkos.h
```cpp
#ifndef __JACOBI_KOKKOS_H
#define __JACOBI_KOKKOS_H

#include <vector>

#include <Kokkos_Core.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy);

#endif  // __JACOBI_KOKKOS_H
```
- jacobi_kokkos.cpp
```cpp
#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    // Place your implementation here
}
```

# Results
## 1_permutations_cxx (102400 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI1|chistov_alexey|0.0672|1|
|3822B1FI1|shulpin_ilya|0.0710|8|
|3822B1FI1|beskhmelnova_kseniya|0.0844|6|
|3822B1FI2|khokhlov_andrey|0.1532|4|
|3822B1FI1|rezantseva_anastasia|0.1600|7|
|3822B1FI1|drozhdinov_dmitriy|0.1613|3|
|3822B1FI3|ekaterina_kozlova|0.1614|8|
|3822B1FI3|kholin_kirill|0.1891|5|
|3822B1FI3|frolova_elizaveta|0.1951|4|
|3822B1FI3|lopatin_ilya|0.2017|1|
|3822B1FI3|lysov_ivan|0.2044|6|
|3822B1FI1|kabalova_valeria|0.2107|2|
|3822B1FI1|solovev_alexey|0.2119|5|
|3822B1FI2|bessonov_egor|0.2228|1|
|3822B1FI2|guseynov_emil|0.2257|3|
|3822B1FI3|sozonov_ilya|0.2314|3|
|3822B1FI1|shurigin_sergey|0.2419|4|
|3822B1FI3|kolodkin_grigorii|0.2419|2|
|3822B1FI2|yasakova_tanya|0.2451|2|
|**REF**|**REF**|**0.2492**|**-**|
|3822B1FI1|mironov_arseniy|0.2517|10|
|3822B1FI1|ionova_ekaterina|0.2542|9|
|3822B1FI3|chizhov_maxim|0.3237|7|

## 2_integral_oneapi (65536 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI1|chistov_alexey|0.4474|3|
|**REF**|**REF**|**0.4723**|**-**|
|3822B1FI1|rezantseva_anastasia|0.5868|5|
|3822B1FI2|yasakova_tanya|0.8381|2|
|3822B1FI3|frolova_elizaveta|0.8388|3|
|3822B1FI1|shurigin_sergey|0.8417|7|
|3822B1FI3|kolodkin_grigorii|0.8418|2|
|3822B1FI1|beskhmelnova_kseniya|0.8425|2|
|3822B1FI1|solovev_alexey|0.8441|4|
|3822B1FI1|ionova_ekaterina|0.8475|9|
|3822B1FI1|kabalova_valeria|0.8483|1|
|3822B1FI3|lysov_ivan|0.8516|5|
|3822B1FI2|guseynov_emil|0.9945|3|
|3822B1FI3|lopatin_ilya|0.9945|1|
|3822B1FI1|drozhdinov_dmitriy|0.9980|6|
|3822B1FI3|ekaterina_kozlova|0.9980|6|
|3822B1FI1|mironov_arseniy|0.9988|10|
|3822B1FI2|khokhlov_andrey|0.9988|4|
|3822B1FI2|bessonov_egor|1.0007|1|
|3822B1FI3|kholin_kirill|1.0007|4|
|3822B1FI1|shulpin_ilya|1.0230|8|

## 3_acc_jacobi_oneapi (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI1|shulpin_ilya|0.2705|7|
|**REF**|**REF**|**0.2749**|**-**|
|3822B1FI3|lysov_ivan|0.2813|5|
|3822B1FI2|bessonov_egor|0.2871|1|
|3822B1FI3|kolodkin_grigorii|0.3235|1|
|3822B1FI3|kholin_kirill|0.3262|4|
|3822B1FI3|frolova_elizaveta|0.3264|2|
|3822B1FI1|shurigin_sergey|0.3290|4|
|3822B1FI1|rezantseva_anastasia|0.3366|6|
|3822B1FI1|solovev_alexey|0.3380|5|
|3822B1FI1|beskhmelnova_kseniya|0.3462|1|
|3822B1FI3|lopatin_ilya|0.3551|3|
|3822B1FI1|chistov_alexey|0.3639|3|
|3822B1FI2|yasakova_tanya|0.3980|2|
|3822B1FI1|drozhdinov_dmitriy|0.5052|2|
|3822B1FI1|mironov_arseniy|TEST FAILED|-|
|3822B1FI1|ionova_ekaterina|TEST FAILED|-|
|3822B1FI3|ekaterina_kozlova|BUILD FAILED|-|
|3822B1FI2|khokhlov_andrey|TEST FAILED|-|

## 4_dev_jacobi_oneapi (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI1|rezantseva_anastasia|0.1838|6|
|3822B1FI1|ionova_ekaterina|0.2077|9|
|3822B1FI1|shulpin_ilya|0.2275|8|
|3822B1FI1|mironov_arseniy|0.2564|7|
|**REF**|**REF**|**0.2701**|**-**|
|3822B1FI2|bessonov_egor|0.2818|1|
|3822B1FI1|solovev_alexey|0.2909|3|
|3822B1FI1|shurigin_sergey|0.2945|5|
|3822B1FI3|lopatin_ilya|0.3083|3|
|3822B1FI3|kolodkin_grigorii|0.3165|1|
|3822B1FI3|kholin_kirill|0.3246|4|
|3822B1FI3|ekaterina_kozlova|0.3271|6|
|3822B1FI3|lysov_ivan|0.3307|5|
|3822B1FI1|drozhdinov_dmitriy|0.3362|4|
|3822B1FI2|yasakova_tanya|0.3389|2|
|3822B1FI2|khokhlov_andrey|0.4715|3|
|3822B1FI3|frolova_elizaveta|0.4842|2|
|3822B1FI1|beskhmelnova_kseniya|0.5692|1|
|3822B1FI1|chistov_alexey|0.6086|2|

## 5_shared_jacobi_oneapi (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI1|chistov_alexey|0.1641|1|
|3822B1FI1|rezantseva_anastasia|0.1869|6|
|3822B1FI1|ionova_ekaterina|0.2024|8|
|3822B1FI1|shulpin_ilya|0.2351|7|
|3822B1FI2|bessonov_egor|0.2362|1|
|3822B1FI1|mironov_arseniy|0.2412|9|
|**REF**|**REF**|**0.2633**|**-**|
|3822B1FI1|shurigin_sergey|0.2827|5|
|3822B1FI3|lysov_ivan|0.2910|4|
|3822B1FI3|kholin_kirill|0.3080|5|
|3822B1FI1|drozhdinov_dmitriy|0.3305|4|
|3822B1FI1|solovev_alexey|0.3317|3|
|3822B1FI1|beskhmelnova_kseniya|0.3483|2|
|3822B1FI2|yasakova_tanya|0.3532|2|
|3822B1FI3|lopatin_ilya|0.3664|3|
|3822B1FI3|kolodkin_grigorii|0.4017|1|
|3822B1FI3|frolova_elizaveta|0.4647|2|
|3822B1FI2|khokhlov_andrey|TEST FAILED|-|
|3822B1FI3|ekaterina_kozlova|BUILD FAILED|-|

## 6_block_gemm_oneapi (3072 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI1|rezantseva_anastasia|0.8756|5|
|3822B1FI3|frolova_elizaveta|0.8798|2|
|3822B1FI3|ekaterina_kozlova|0.8832|5|
|3822B1FI1|beskhmelnova_kseniya|0.8884|1|
|3822B1FI2|bessonov_egor|0.8908|2|
|3822B1FI1|solovev_alexey|0.8935|3|
|3822B1FI2|yasakova_tanya|0.8962|1|
|3822B1FI3|lopatin_ilya|0.8981|3|
|3822B1FI1|drozhdinov_dmitriy|0.9047|6|
|3822B1FI3|kholin_kirill|0.9087|4|
|3822B1FI1|shurigin_sergey|0.9134|4|
|**REF**|**REF**|**0.9144**|**-**|
|3822B1FI1|chistov_alexey|0.9184|2|
|3822B1FI1|ionova_ekaterina|0.9200|8|
|3822B1FI3|lysov_ivan|0.9502|6|
|3822B1FI3|kolodkin_grigorii|2.1427|1|
|3822B1FI1|shulpin_ilya|3.5330|7|
|3822B1FI2|khokhlov_andrey|RUN FAILED|-|

## 7_mkl_gemm_oneapi (3072 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI3|ekaterina_kozlova|0.3985|4|
|3822B1FI1|shurigin_sergey|0.4019|7|
|3822B1FI1|rezantseva_anastasia|0.4031|4|
|3822B1FI1|beskhmelnova_kseniya|0.4091|2|
|3822B1FI1|shulpin_ilya|0.4137|5|
|3822B1FI2|yasakova_tanya|0.4255|1|
|3822B1FI1|drozhdinov_dmitriy|0.4314|6|
|3822B1FI3|lopatin_ilya|0.4338|3|
|3822B1FI1|solovev_alexey|0.4340|3|
|3822B1FI2|bessonov_egor|0.4369|2|
|3822B1FI1|chistov_alexey|0.4392|1|
|**REF**|**REF**|**0.4392**|**-**|
|3822B1FI3|kolodkin_grigorii|0.4450|1|
|3822B1FI3|frolova_elizaveta|0.4468|2|
|3822B1FI3|lysov_ivan|0.4583|5|
|3822B1FI1|ionova_ekaterina|TEST FAILED|-|

## 8_integral_kokkos (65536 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI3|lopatin_ilya|0.0005|3|
|3822B1FI1|solovev_alexey|0.0006|3|
|3822B1FI1|chistov_alexey|0.0009|1|
|3822B1FI1|rezantseva_anastasia|0.0010|5|
|3822B1FI1|beskhmelnova_kseniya|0.0011|2|
|3822B1FI2|yasakova_tanya|0.0022|1|
|**REF**|**REF**|**0.3629**|**-**|
|3822B1FI3|frolova_elizaveta|2.2278|2|
|3822B1FI3|kolodkin_grigorii|2.3206|1|
|3822B1FI1|ionova_ekaterina|2.3228|6|
|3822B1FI1|drozhdinov_dmitriy|2.3230|7|
|3822B1FI2|bessonov_egor|2.3230|2|
|3822B1FI1|shulpin_ilya|2.9876|4|
|3822B1FI1|shurigin_sergey|2.9995|8|
|3822B1FI3|ekaterina_kozlova|RUN FAILED|-|

## 9_jacobi_kokkos (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3822B1FI1|ionova_ekaterina|0.2522|7|
|3822B1FI1|shulpin_ilya|0.2572|6|
|**REF**|**REF**|**0.2679**|**-**|
|3822B1FI3|lopatin_ilya|0.3165|3|
|3822B1FI1|shurigin_sergey|0.3170|5|
|3822B1FI2|bessonov_egor|0.3179|2|
|3822B1FI1|chistov_alexey|0.3202|2|
|3822B1FI1|beskhmelnova_kseniya|0.3214|1|
|3822B1FI3|ekaterina_kozlova|0.3293|4|
|3822B1FI3|frolova_elizaveta|0.3331|2|
|3822B1FI1|rezantseva_anastasia|0.3360|4|
|3822B1FI2|yasakova_tanya|0.3641|1|
|3822B1FI3|kolodkin_grigorii|0.3787|1|
|3822B1FI1|solovev_alexey|0.3892|3|

# Tasks Done
## 3822B1FI1
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1FI1|beskhmelnova_kseniya|**9/9**|**533**|
|3822B1FI1|chistov_alexey|**9/9**|**541**|
|3822B1FI1|drozhdinov_dmitriy|8/9|441|
|3822B1FI1|ionova_ekaterina|7/9|372|
|3822B1FI1|kabalova_valeria|2/9|116|
|3822B1FI1|mironov_arseniy|4/9|201|
|3822B1FI1|rezantseva_anastasia|**9/9**|**522**|
|3822B1FI1|shulpin_ilya|**9/9**|**493**|
|3822B1FI1|shurigin_sergey|**9/9**|**503**|
|3822B1FI1|solovev_alexey|**9/9**|**516**|

Passed: 6

## 3822B1FI2
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1FI2|bessonov_egor|**9/9**|**566**|
|3822B1FI2|guseynov_emil|2/9|121|
|3822B1FI2|khokhlov_andrey|3/9|180|
|3822B1FI2|yasakova_tanya|**9/9**|**563**|

Passed: 2

## 3822B1FI3
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3822B1FI3|chizhov_maxim|1/9|51|
|3822B1FI3|ekaterina_kozlova|6/9|348|
|3822B1FI3|frolova_elizaveta|**9/9**|**544**|
|3822B1FI3|kholin_kirill|6/9|350|
|3822B1FI3|kolodkin_grigorii|**9/9**|**550**|
|3822B1FI3|lopatin_ilya|**9/9**|**547**|
|3822B1FI3|lysov_ivan|7/9|401|
|3822B1FI3|sozonov_ilya|1/9|57|

Passed: 3

**Total Passed: 11**

---
*Maximum Score: 576 (64 per task)*
