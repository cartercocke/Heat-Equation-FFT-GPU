/**
 * @file gpu_heat_equation.cu
 * @brief These are the functions used to solve the heat equation on the GPU
 *
 * @author Carter Cocke
 *
 * @date 06/03/2023
 */
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

#include "helper_structs.h"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

/**
 * @brief Check the return value of a CUDA call
 *
 * @param code Return value of the CUDA call
 * @param file File name of the CUDA call
 * @param line Line number of the CUDA call
 */
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/**
 * @brief Check the return value of a FFT call
 *
 * @param errval Error value returned by the FFT call
 */
void gpuFFTchk(int errval)
{
    if (errval != CUFFT_SUCCESS)
    {
        std::cerr << "Failed FFT call, error code " << errval << std::endl;
    }
}

/**
 * @brief Compute the spatial frequency vector \f$k = (k_x, k_y)\f$
 *
 * @param i Grid index in x direction
 * @param j Grid index in y direction
 * @param grid Grid struct containing the grid size
 * @param kx Spatial frequency in x direction
 * @param ky Spatial frequency in y direction
 */
__device__ double spatial_frequency(const int i,
                                    const int j,
                                    const grid_size grid,
                                    double *kx,
                                    double *ky)
{
    if (i < grid.Nx / 2)
    {
        *kx = 2. * M_PI * i / grid.Nx;
    }
    else
    {
        *kx = 2. * M_PI * (i - grid.Nx) / grid.Nx;
    }
    if (j < grid.Ny / 2)
    {
        *ky = 2. * M_PI * j / grid.Ny;
    }
    else
    {
        *ky = 2. * M_PI * (j - grid.Ny) / grid.Ny;
    }
}

/**
 * @brief Compute the square of the norm of the spatial frequency vector \f$\|k\|_2^2\f$
 *
 * @param i Grid index in x direction
 * @param j Grid index in y direction
 * @param grid Grid struct containing the grid size
 * @return double squared norm of the spatial frequency vector
 */
__device__ double spatial_freq_square_norm(const int i,
                                           const int j,
                                           const grid_size grid)
{
    double kx, ky;
    spatial_frequency(i, j, grid, &kx, &ky);
    return (kx * kx + ky * ky) / grid.Ntot;
}

/**
 * @brief Compute the Laplacian of the temperature field in Fourier space
 *
 * @param temperature_fourier Temperature field in Fourier space
 * @param grid Grid struct containing the grid size
 */
__global__ void cudaFourierLaplacianKernel(cufftDoubleComplex *temperature_fourier,
                                           const grid_size grid,
                                           const material material)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    double k_norm2 = spatial_freq_square_norm(i, j, grid);
    temperature_fourier[j * grid.Ny + i].x *= k_norm2 * material.k;
    temperature_fourier[j * grid.Ny + i].y *= k_norm2 * material.k;
}

/**
 * @brief Compute the gradient of the temperature field in Fourier space
 *
 * @param fourier_grid_1 First fourier grid for x component of gradient
 * @param fourier_grid_2 Second fourier grid for y component of gradient
 * @param grid Grid struct containing the grid size
 */
__global__ void cudaFourierGradientKernel(cufftDoubleComplex *fourier_grid_1,
                                          cufftDoubleComplex *fourier_grid_2,
                                          const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    double kx, ky;
    spatial_frequency(i, j, grid, &kx, &ky);
    fourier_grid_2[j * grid.Ny + i].x = fourier_grid_1[j * grid.Ny + i].x * ky / grid.Ntot;
    fourier_grid_2[j * grid.Ny + i].y = fourier_grid_1[j * grid.Ny + i].y * ky / grid.Ntot;
    fourier_grid_1[j * grid.Ny + i].x *= kx / grid.Ntot;
    fourier_grid_1[j * grid.Ny + i].y *= kx / grid.Ntot;
}

/**
 * @brief Compute the divergence of a vector field in Fourier space
 *
 * @param fourier_grid_1 FFT of first component of the vector field
 * @param fourier_grid_2 FFT of second component of the vector field
 * @param grid Grid struct containing the grid size
 */
__global__ void cudaFourierDivergenceKernel(cufftDoubleComplex *fourier_grid_1,
                                            cufftDoubleComplex *fourier_grid_2,
                                            const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    double kx, ky;
    spatial_frequency(i, j, grid, &kx, &ky);
    fourier_grid_1[j * grid.Ny + i].x *= kx / grid.Ntot;
    fourier_grid_1[j * grid.Ny + i].y *= kx / grid.Ntot;
    fourier_grid_1[j * grid.Ny + i].x += fourier_grid_2[j * grid.Ny + i].x * ky / grid.Ntot;
    fourier_grid_1[j * grid.Ny + i].y += fourier_grid_2[j * grid.Ny + i].y * ky / grid.Ntot;
}

/**
 * @brief Multiply the gradient of the temperature field by the thermal conductivity
 *
 * @param fourier_grid_1 FFT of first component of the vector field
 * @param fourier_grid_2 FFT of second component of the vector field
 * @param grid Grid struct containing the grid size
 * @param material Material struct containing the thermal conductivity
 */
__global__ void cudaKGridPointwise(cufftDoubleComplex *fourier_grid_1,
                                   cufftDoubleComplex *fourier_grid_2,
                                   const double *dev_k_field,
                                   const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    const double k = dev_k_field[j * grid.Ny + i];

    fourier_grid_1[j * grid.Ny + i].x *= k;
    fourier_grid_1[j * grid.Ny + i].y *= k;
    fourier_grid_2[j * grid.Ny + i].x *= k;
    fourier_grid_2[j * grid.Ny + i].y *= k;
}

/**
 * @brief Copy a double grid to a complex grid
 *
 * @param double_grid Double grid to copy
 * @param fft_grid Complex grid to copy to
 * @param grid Grid struct containing the grid size
 */
__global__ void cudaCopyDoubleToComplexKernel(const double *double_grid,
                                              cufftDoubleComplex *fft_grid,
                                              const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    fft_grid[j * grid.Ny + i].x = double_grid[j * grid.Ny + i];
    fft_grid[j * grid.Ny + i].y = 0.0;
}

/**
 * @brief Compute the function evaluation q/rho*cp - alpha*laplacian(T)
 *
 * @param dev_f_eval Final evaluation of the function RHS
 * @param dev_heat_source Device heat source field to evaluate the function at
 * @param dev_temperature_laplacian Laplacian of the temperature field in Fourier space
 * @param grid Grid specifications
 * @param material Material specifications
 */
__global__ void cudaFinalEvaluationKernel(double *dev_f_eval,
                                          const double *dev_heat_source,
                                          const cufftDoubleComplex *dev_temperature_laplacian,
                                          const grid_size grid,
                                          const material material)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    dev_f_eval[j * grid.Ny + i] = (dev_heat_source[j * grid.Ny + i] - dev_temperature_laplacian[j * grid.Ny + i].x) / (material.rho * material.cp);
}

/**
 * @brief Compute the RHS of the heat equation with a given temperature field and heat source
 *
 * @param dimGrid Grid dimensions
 * @param dimBlock Block dimensions
 * @param dev_f_eval Final evaluation of the function RHS
 * @param dev_temperature Device temperature field to evaluate the function at
 * @param dev_heat_source Device heat source field to evaluate the function at
 * @param dev_fourier_grid Device FFTW complex grid
 * @param dev_real_grid Device FFTW real grid
 * @param plan FFTW forward plan
 * @param grid Grid specifications
 * @param material Material specifications
 */
void callCudaEvaluateRHS(const dim3 dimGrid,
                         const dim3 dimBlock,
                         double *dev_f_eval,
                         const double *dev_temperature,
                         const double *dev_heat_source,
                         cufftDoubleComplex *dev_fourier_grid,
                         cufftDoubleComplex *dev_real_grid,
                         cufftHandle plan,
                         const grid_size grid,
                         const material material)
{
    // Fill temperature FFT grid
    cudaCopyDoubleToComplexKernel<<<dimGrid, dimBlock>>>(dev_temperature, dev_real_grid, grid);
    gpuErrchk(cudaPeekAtLastError());

    // FFT(temperature)
    gpuFFTchk(cufftExecZ2Z(plan, dev_real_grid, dev_fourier_grid, CUFFT_FORWARD));

    // Compute laplacian(T_hat) in Fourier space
    cudaFourierLaplacianKernel<<<dimGrid, dimBlock>>>(dev_fourier_grid, grid, material);
    gpuErrchk(cudaPeekAtLastError());

    // IFFT(laplacian(T_hat))
    gpuFFTchk(cufftExecZ2Z(plan, dev_fourier_grid, dev_real_grid, CUFFT_INVERSE));

    // Compute function evaluation q/rho*cp - alpha*laplacian(T)
    cudaFinalEvaluationKernel<<<dimGrid, dimBlock>>>(dev_f_eval, dev_heat_source, dev_real_grid, grid, material);
    gpuErrchk(cudaPeekAtLastError());
}

/**
 * @brief Compute the RHS of the heat equation with a given temperature field and heat source
 *
 * @param dimGrid Grid dimensions
 * @param dimBlock Block dimensions
 * @param dev_f_eval Final evaluation of the function RHS
 * @param dev_temperature Device temperature field to evaluate the function at
 * @param dev_heat_source Device heat source field to evaluate the function at
 * @param dev_fourier_grid_1 Device FFTW complex grid
 * @param dev_fourier_grid_2 Device FFTW complex grid
 * @param dev_real_grid_1 Device FFTW real grid
 * @param dev_real_grid_2 Device FFTW real grid
 * @param plan FFTW forward plan
 * @param grid Grid specifications
 * @param material Material specifications
 */
void callCudaEvaluateRHSSpatiallyVarying(const dim3 dimGrid,
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
                                         const material material)
{
    // Fill temperature FFT grid
    cudaCopyDoubleToComplexKernel<<<dimGrid, dimBlock>>>(dev_temperature, dev_real_grid_1, grid);
    gpuErrchk(cudaPeekAtLastError());

    // FFT(temperature)
    gpuFFTchk(cufftExecZ2Z(plan, dev_real_grid_1, dev_fourier_grid_1, CUFFT_FORWARD));

    // Compute grad(T_hat) in Fourier space
    cudaFourierGradientKernel<<<dimGrid, dimBlock>>>(dev_fourier_grid_1, dev_fourier_grid_2, grid);
    gpuErrchk(cudaPeekAtLastError());

    // IFFT(grad(T_hat))
    gpuFFTchk(cufftExecZ2Z(plan, dev_fourier_grid_1, dev_real_grid_1, CUFFT_INVERSE));
    gpuFFTchk(cufftExecZ2Z(plan, dev_fourier_grid_2, dev_real_grid_2, CUFFT_INVERSE));

    // Compute k * grad(T)
    cudaKGridPointwise<<<dimGrid, dimBlock>>>(dev_real_grid_1, dev_real_grid_2, dev_k_field, grid);
    gpuErrchk(cudaPeekAtLastError());

    // FFT(k * grad(T))
    gpuFFTchk(cufftExecZ2Z(plan, dev_real_grid_1, dev_fourier_grid_1, CUFFT_FORWARD));
    gpuFFTchk(cufftExecZ2Z(plan, dev_real_grid_2, dev_fourier_grid_2, CUFFT_FORWARD));

    // Compute div(k * grad(T)) in Fourier space
    cudaFourierDivergenceKernel<<<dimGrid, dimBlock>>>(dev_fourier_grid_1, dev_fourier_grid_2, grid);
    gpuErrchk(cudaPeekAtLastError());

    // IFFT(div(k * grad(T)))
    gpuFFTchk(cufftExecZ2Z(plan, dev_fourier_grid_1, dev_real_grid_1, CUFFT_INVERSE));

    // Compute function evaluation q/rho*cp - div(k * grad(T))
    cudaFinalEvaluationKernel<<<dimGrid, dimBlock>>>(dev_f_eval, dev_heat_source, dev_real_grid_1, grid, material);
    gpuErrchk(cudaPeekAtLastError());
}

/**
 * @brief Wrapper function for evaluating the RHS of the heat equation
 *
 * @copydoc callCudaEvaluateRHSSpatiallyVarying
 * @param spatial_variation Type of spatial variation to use
 */
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
                                 const std::string spatial_variation)
{
    if (spatial_variation == "constant")
    {
        callCudaEvaluateRHS(dimGrid, dimBlock, dev_f_eval, dev_temperature, dev_heat_source,
                            dev_fourier_grid_1, dev_real_grid_1, plan, grid, material);
    }
    else if (spatial_variation == "half" || spatial_variation == "checkerboard")
    {
        callCudaEvaluateRHSSpatiallyVarying(dimGrid, dimBlock, dev_f_eval, dev_temperature, dev_heat_source, dev_k_field,
                                            dev_fourier_grid_1, dev_fourier_grid_2, dev_real_grid_1, dev_real_grid_2,
                                            plan, grid, material);
    }
    else
    {
        std::cout << "Invalid spatial_variation defined" << std::endl;
        exit(1);
    }
}

/**
 * @brief Compute C = A + alpha * B
 *
 * @param dev_grid1 Grid A
 * @param dev_grid2 Grid B
 * @param dev_result Grid C
 * @param alpha Weight of B
 * @param grid Grid specifications
 */
__global__ void cudaWeightedGridAddKernel(const double *dev_vec1,
                                          const double *dev_vec2,
                                          double *dev_result,
                                          const double alpha,
                                          const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    dev_result[j * grid.Ny + i] = dev_vec1[j * grid.Ny + i] + alpha * dev_vec2[j * grid.Ny + i];
}

/**
 * @copydoc cudaEulerExplicitKernel
 *
 * @param dimGrid Grid dimensions
 * @param dimBlock Block dimensions
 */
void callCudaWeightedGridAddKernel(const dim3 dimGrid,
                                   const dim3 dimBlock,
                                   const double *dev_vec1,
                                   const double *dev_vec2,
                                   double *dev_result,
                                   const double alpha,
                                   const grid_size grid)
{
    cudaWeightedGridAddKernel<<<dimGrid, dimBlock>>>(dev_vec1, dev_vec2, dev_result, alpha, grid);
    gpuErrchk(cudaPeekAtLastError());
}

/**
 * @brief Downcast a double grid to a float grid
 *
 * @param dev_double_grid Double grid to downcast
 * @param dev_float_grid Float grid to downcast to
 * @param grid Grid specifications
 */
__global__ void cudaCastDoubleToFloatKernel(const double *dev_double_grid,
                                            float *dev_float_grid,
                                            const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    dev_float_grid[j * grid.Ny + i] = (float)dev_double_grid[j * grid.Ny + i];
}

/**
 * @copydoc cudaCastDoubleToFloatKernel
 *
 * @param dimGrid Grid dimensions
 * @param dimBlock Block dimensions
 */
void callCudaCastDoubleToFloatKernel(const dim3 dimGrid,
                                     const dim3 dimBlock,
                                     const double *dev_double_grid,
                                     float *dev_float_grid,
                                     const grid_size grid)
{
    cudaCastDoubleToFloatKernel<<<dimGrid, dimBlock>>>(dev_double_grid, dev_float_grid, grid);
    gpuErrchk(cudaPeekAtLastError());
}

/**
 * @brief Explicit Euler time step update
 *
 * @param temperature Temperature field at time t (updated to t+dt)
 * @param f_eval Function evaluation at time t
 * @param grid Grid struct containing the grid size
 * @param dt Time increment dt
 */
__global__ void cudaEulerExplicitKernel(double *temperature,
                                        const double *f_eval,
                                        const double dt,
                                        const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    temperature[j * grid.Ny + i] += dt * f_eval[j * grid.Ny + i];
}

/**
 * @copydoc cudaEulerExplicitKernel
 *
 * @param dimGrid Grid dimensions
 * @param dimBlock Block dimensions
 */
void callCudaEulerExplicitKernel(const dim3 dimGrid,
                                 const dim3 dimBlock,
                                 double *temperature,
                                 const double *f_eval,
                                 const double dt,
                                 const grid_size grid)
{
    cudaEulerExplicitKernel<<<dimGrid, dimBlock>>>(temperature, f_eval, dt, grid);
    gpuErrchk(cudaPeekAtLastError());
}

/**
 * @brief RK4 time step update
 *
 * @param temperature Temperature field at time t (updated to time t + dt)
 * @param k1 Evaluation of the function at (t, T)
 * @param k2 Evaluation of the function at (t + dt/2, T + dt/2 * k1)
 * @param k3 Evaluation of the function at (t + dt/2, T + dt/2 * k2)
 * @param k4 Evaluation of the function at (t + dt, T + dt * k3)
 * @param dt Time increment dt
 * @param grid Grid struct containing the grid size
 */
__global__ void cudaRK4ExplicitKernel(double *temperature,
                                      const double *k1,
                                      const double *k2,
                                      const double *k3,
                                      const double *k4,
                                      const double dt,
                                      const grid_size grid)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    temperature[j * grid.Ny + i] += dt / 6. * (k1[j * grid.Ny + i] + 2. * k2[j * grid.Ny + i] + 2. * k3[j * grid.Ny + i] + k4[j * grid.Ny + i]);
}

/**
 * @copydoc cudaRK4ExplicitKernel
 *
 * @param dimGrid Grid dimensions
 * @param dimBlock Block dimensions
 */
void callCudaRK4ExplicitKernel(const dim3 dimGrid,
                               const dim3 dimBlock,
                               double *temperature,
                               const double *k1,
                               const double *k2,
                               const double *k3,
                               const double *k4,
                               const double dt,
                               const grid_size grid)
{
    cudaRK4ExplicitKernel<<<dimGrid, dimBlock>>>(temperature, k1, k2, k3, k4, dt, grid);
    gpuErrchk(cudaPeekAtLastError());
}