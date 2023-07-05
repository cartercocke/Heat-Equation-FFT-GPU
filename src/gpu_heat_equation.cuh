#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include "helper_structs.h"

void gpuAssert(cudaError_t code,
               const char *file,
               int line);

void gpuFFTchk(int errval);

void callCudaEulerExplicitKernel(const dim3 dimGrid,
                                 const dim3 dimBlock,
                                 double *temperature,
                                 const double *f_eval,
                                 const double dt,
                                 const grid_size grid);

void callCudaRK4ExplicitKernel(const dim3 dimGrid,
                               const dim3 dimBlock,
                               double *temperature,
                               const double *k1,
                               const double *k2,
                               const double *k3,
                               const double *k4,
                               const double dt,
                               const grid_size grid);

void cudaEvaluateFunctionWrapper(const dim3 dimGrid,
                                 const dim3 dimBlock,
                                 double *dev_f_eval,
                                 const double *dev_temperature,
                                 const double *dev_heat_source,
                                 const double *dev_k_field,
                                 cufftDoubleComplex *dev_fourier_grid_1,
                                 cufftDoubleComplex *dev_fourier_grid_2,
                                 cufftDoubleComplex *dev_real_grid_1,
                                 cufftDoubleComplex *dev_real_grid_2,
                                 cufftHandle plan,
                                 const grid_size grid,
                                 const material material,
                                 const std::string spatial_variation);

void callCudaWeightedGridAddKernel(const dim3 dimGrid,
                                   const dim3 dimBlock,
                                   const double *dev_vec1,
                                   const double *dev_vec2,
                                   double *dev_result,
                                   const double alpha,
                                   const grid_size grid);

void callCudaCastDoubleToFloatKernel(const dim3 dimGrid,
                                     const dim3 dimBlock,
                                     const double *dev_double_grid,
                                     float *dev_float_grid,
                                     const grid_size grid);