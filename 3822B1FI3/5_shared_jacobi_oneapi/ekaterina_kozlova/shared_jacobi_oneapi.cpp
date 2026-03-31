#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <algorithm>

const int MAX_ITERATION_LIMIT = 20000;

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device& device) {
    
    size_t matrixSize = b.size();
    sycl::queue computeQueue(device);

    std::vector<float> oldSolution(matrixSize, 0.0f);     
    std::vector<float> newSolution(matrixSize, 0.0f);     
    std::vector<float> errorVector(matrixSize, 0.0f);     

    {
        sycl::buffer<float, 1> matrixBuffer(a.data(), sycl::range<1>(matrixSize * matrixSize));
        sycl::buffer<float, 1> rhsBuffer(b.data(), sycl::range<1>(matrixSize));
        sycl::buffer<float, 1> oldBuffer(oldSolution.data(), sycl::range<1>(matrixSize));
        sycl::buffer<float, 1> newBuffer(newSolution.data(), sycl::range<1>(matrixSize));
        sycl::buffer<float, 1> errorBuffer(errorVector.data(), sycl::range<1>(matrixSize));

        float currentError = accuracy + 1.0f;
        int iterationStep = 0;

        while (currentError > accuracy && iterationStep < MAX_ITERATION_LIMIT) {
            iterationStep++;

            computeQueue.submit([&](sycl::handler& commandHandler) {
                auto matrixAcc = matrixBuffer.get_access<sycl::access::mode::read>(commandHandler);
                auto rhsAcc = rhsBuffer.get_access<sycl::access::mode::read>(commandHandler);
                auto oldAcc = oldBuffer.get_access<sycl::access::mode::read>(commandHandler);
                auto newAcc = newBuffer.get_access<sycl::access::mode::write>(commandHandler);

                commandHandler.parallel_for(sycl::range<1>(matrixSize), [=](sycl::id<1> position) {
                    float offDiagonalSum = 0.0f;
                    float diagValue = 0.0f;
                    size_t rowIndex = position[0];
                    
                    for (size_t colIndex = 0; colIndex < matrixSize; ++colIndex) {
                        float matrixElement = matrixAcc[rowIndex * matrixSize + colIndex];
                        if (colIndex == rowIndex) {
                            diagValue = matrixElement;
                        } else {
                            offDiagonalSum += matrixElement * oldAcc[colIndex];
                        }
                    }
                    newAcc[rowIndex] = (rhsAcc[rowIndex] - offDiagonalSum) / diagValue;
                });
            });
            
            computeQueue.submit([&](sycl::handler& commandHandler) {
                auto oldAcc = oldBuffer.get_access<sycl::access::mode::read>(commandHandler);
                auto newAcc = newBuffer.get_access<sycl::access::mode::read>(commandHandler);
                auto errorAcc = errorBuffer.get_access<sycl::access::mode::write>(commandHandler);
                
                commandHandler.parallel_for(sycl::range<1>(matrixSize), [=](sycl::id<1> idx) {
                    errorAcc[idx[0]] = std::fabs(newAcc[idx[0]] - oldAcc[idx[0]]);
                });
            });

            computeQueue.submit([&](sycl::handler& commandHandler) {
                auto oldAcc = oldBuffer.get_access<sycl::access::mode::write>(commandHandler);
                auto newAcc = newBuffer.get_access<sycl::access::mode::read>(commandHandler);
                
                commandHandler.parallel_for(sycl::range<1>(matrixSize), [=](sycl::id<1> idx) {
                    oldAcc[idx[0]] = newAcc[idx[0]];
                });
            }).wait();

            auto hostErrorData = errorVector.data();
            currentError = 0.0f;
            for (size_t i = 0; i < matrixSize; ++i) {
                currentError = std::max(currentError, hostErrorData[i]);
            }
        }
    }

    return oldSolution;
}
