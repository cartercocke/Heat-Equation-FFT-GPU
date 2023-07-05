/**
 * @file gpu_demo.cpp
 * @brief This is the main file for the GPU demo
 *
 * @author Carter Cocke
 *
 * @date 06/03/2023
 */
#include <iostream>
#include <math.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "helper_structs.h"
#include "gpu_heat_equation.cuh"
#include "example_fields.h"
#include "solution_io.h"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

int main()
{
    // Read input file
    int Nx, Ny, timesteps;
    double time_increment;
    grid_size grid;
    material material;
    std::string method, spatial_variation, initial_temperature, heat_generation;

    std::string in_filename = "src/input_file.txt";
    read_input(in_filename, &Nx, &Ny, &timesteps, &time_increment, &method,
               &spatial_variation, &initial_temperature, &heat_generation, &grid, &material);

    // Plan 2D FFTs (Initialize plan first per documentation)
    cufftHandle plan;
    gpuFFTchk(cufftPlan2d(&plan, Nx, Ny, CUFFT_Z2Z));

    cufftDoubleComplex *dev_real_grid_1, *dev_real_grid_2, *dev_fourier_grid_1, *dev_fourier_grid_2;
    gpuErrchk(cudaMalloc((void **)&dev_real_grid_1, sizeof(cufftDoubleComplex) * Nx * Ny));
    gpuErrchk(cudaMalloc((void **)&dev_fourier_grid_1, sizeof(cufftDoubleComplex) * Nx * Ny));
    gpuErrchk(cudaMalloc((void **)&dev_real_grid_2, sizeof(cufftDoubleComplex) * Nx * Ny));
    gpuErrchk(cudaMalloc((void **)&dev_fourier_grid_2, sizeof(cufftDoubleComplex) * Nx * Ny));

    // Solution array
    float *solution = (float *)malloc(sizeof(float) * Nx * Ny * timesteps);
    double *temperature = (double *)malloc(sizeof(double) * Nx * Ny);
    double *k_field = (double *)malloc(sizeof(double) * Nx * Ny);
    double *heat_source = (double *)malloc(sizeof(double) * Nx * Ny);

    float *dev_temperature_float;
    double *dev_temperature, *dev_heat_source, *dev_k_field, *dev_k1, *dev_k2, *dev_k3, *dev_k4, *dev_inter_temperature;
    gpuErrchk(cudaMalloc((void **)&dev_temperature, sizeof(double) * Nx * Ny));
    gpuErrchk(cudaMalloc((void **)&dev_temperature_float, sizeof(float) * Nx * Ny));
    gpuErrchk(cudaMalloc((void **)&dev_k_field, sizeof(double) * Nx * Ny));
    gpuErrchk(cudaMalloc((void **)&dev_heat_source, sizeof(double) * Nx * Ny));
    gpuErrchk(cudaMalloc((void **)&dev_k1, sizeof(double) * Nx * Ny));
    if (method == "rk4")
    {
        gpuErrchk(cudaMalloc((void **)&dev_k2, sizeof(double) * Nx * Ny));
        gpuErrchk(cudaMalloc((void **)&dev_k3, sizeof(double) * Nx * Ny));
        gpuErrchk(cudaMalloc((void **)&dev_k4, sizeof(double) * Nx * Ny));
        gpuErrchk(cudaMalloc((void **)&dev_inter_temperature, sizeof(double) * Nx * Ny));
    }

    // Fill initial temperature field and define heat source
    heat_transfer_field(k_field, grid, material, spatial_variation);
    temperature_field(temperature, grid, initial_temperature);
    heat_source_field(heat_source, grid, heat_generation);

    // Copy initial temperature field and heat source to device asynchronously
    gpuErrchk(cudaMemcpyAsync(dev_k_field, k_field, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyAsync(dev_temperature, temperature, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyAsync(dev_heat_source, heat_source, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice));

    // Fill initial solution array
    for (int i = 0; i < Nx * Ny; i++)
        solution[i] = temperature[i];

    // Define CUDA grid and block sizes
    const dim3 dimBlock(32, 32);
    const dim3 dimGrid(Nx / dimBlock.x, Ny / dimBlock.y);

    // Timing
    float runtime_ms;
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Time evolution
    std::cout << "Running GPU demo" << std::endl;
    cudaEventRecord(start);
    for (int t = 1; t < timesteps; t++)
    {
        if (method == "euler")
        {
            cudaEvaluateFunctionWrapper(dimGrid, dimBlock, dev_k1, dev_temperature, dev_heat_source, dev_k_field,
                                        dev_fourier_grid_1, dev_fourier_grid_2, dev_real_grid_1, dev_real_grid_2,
                                        plan, grid, material, spatial_variation);

            // Integrate in time
            callCudaEulerExplicitKernel(dimGrid, dimBlock, dev_temperature, dev_k1, time_increment, grid);
        }
        else if (method == "rk4")
        {
            // Compute K1
            cudaEvaluateFunctionWrapper(dimGrid, dimBlock, dev_k1, dev_temperature, dev_heat_source, dev_k_field,
                                        dev_fourier_grid_1, dev_fourier_grid_2, dev_real_grid_1, dev_real_grid_2,
                                        plan, grid, material, spatial_variation);

            // Compute K2
            callCudaWeightedGridAddKernel(dimGrid, dimBlock, dev_temperature, dev_k1, dev_inter_temperature, 0.5 * time_increment, grid);

            cudaEvaluateFunctionWrapper(dimGrid, dimBlock, dev_k2, dev_inter_temperature, dev_heat_source, dev_k_field,
                                        dev_fourier_grid_1, dev_fourier_grid_2, dev_real_grid_1, dev_real_grid_2,
                                        plan, grid, material, spatial_variation);

            // Compute K3
            callCudaWeightedGridAddKernel(dimGrid, dimBlock, dev_temperature, dev_k2, dev_inter_temperature, 0.5 * time_increment, grid);

            cudaEvaluateFunctionWrapper(dimGrid, dimBlock, dev_k3, dev_inter_temperature, dev_heat_source, dev_k_field,
                                        dev_fourier_grid_1, dev_fourier_grid_2, dev_real_grid_1, dev_real_grid_2,
                                        plan, grid, material, spatial_variation);

            // Compute K4
            callCudaWeightedGridAddKernel(dimGrid, dimBlock, dev_temperature, dev_k3, dev_inter_temperature, time_increment, grid);

            cudaEvaluateFunctionWrapper(dimGrid, dimBlock, dev_k4, dev_inter_temperature, dev_heat_source, dev_k_field,
                                        dev_fourier_grid_1, dev_fourier_grid_2, dev_real_grid_1, dev_real_grid_2,
                                        plan, grid, material, spatial_variation);

            // Integrate in time
            callCudaRK4ExplicitKernel(dimGrid, dimBlock, dev_temperature, dev_k1, dev_k2, dev_k3, dev_k4, time_increment, grid);
        }
        else
        {
            std::cout << "Invalid method" << std::endl;
            return 1;
        }

        // Downcast double to float and copy to host asynchronously so we can start the next iteration
        // It turns out this is much faster than copying the double array (non-asynchronously) to the host
        // and then downcasting on the CPU
        cudaDeviceSynchronize(); // Make sure the previous iteration's memcpyAsync is complete
        callCudaCastDoubleToFloatKernel(dimGrid, dimBlock, dev_temperature, dev_temperature_float, grid);
        gpuErrchk(cudaMemcpyAsync(&solution[t * Nx * Ny], dev_temperature_float, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runtime_ms, start, stop);
    std::cout << "Elapsed time: " << runtime_ms << " ms" << std::endl;

    // Save solution to binary file
    std::string filename = "bin/solution_gpu.bin";
    write_solution(filename, solution, Nx, Ny, timesteps);

    // Free memory
    gpuFFTchk(cufftDestroy(plan));
    gpuErrchk(cudaFree(dev_real_grid_1));
    gpuErrchk(cudaFree(dev_fourier_grid_1));
    gpuErrchk(cudaFree(dev_temperature));
    gpuErrchk(cudaFree(dev_temperature_float));
    gpuErrchk(cudaFree(dev_heat_source));
    gpuErrchk(cudaFree(dev_k1));
    if (method == "rk4")
    {
        gpuErrchk(cudaFree(dev_k2));
        gpuErrchk(cudaFree(dev_k3));
        gpuErrchk(cudaFree(dev_k4));
        gpuErrchk(cudaFree(dev_inter_temperature));
    }
    free(temperature);
    free(heat_source);
    free(solution);
    std::cout << "Done!" << std::endl;
}
