/**
 * @file cpu_demo.cpp
 * @brief This is the main file for the CPU demo
 *
 * @author Carter Cocke
 *
 * @date 06/03/2023
 */
#include <iostream>
#include <math.h>
#include <chrono>

#include <fftw3.h>
#include "helper_structs.h"
#include "example_fields.h"
#include "solution_io.h"
#include "cpu_heat_equation.h"

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

    // Plan 2D FFTs
    fftw_complex *real_grid_1, *real_grid_2, *fourier_grid_1, *fourier_grid_2;
    fftw_plan forward_plan, backward_plan;

    real_grid_1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fourier_grid_1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    real_grid_2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fourier_grid_2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);

    forward_plan = fftw_plan_dft_2d(Nx, Ny, real_grid_1, fourier_grid_1,
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    backward_plan = fftw_plan_dft_2d(Nx, Ny, fourier_grid_1, real_grid_1,
                                     FFTW_BACKWARD, FFTW_ESTIMATE);

    // Solution array
    float *solution = (float *)malloc(sizeof(float) * Nx * Ny * timesteps);
    double *temperature = (double *)malloc(sizeof(double) * Nx * Ny);
    double *k_field = (double *)malloc(sizeof(double) * Nx * Ny);
    double *heat_source = (double *)malloc(sizeof(double) * Nx * Ny);
    double *k1 = (double *)malloc(sizeof(double) * Nx * Ny);
    double *k2, *k3, *k4, *inter_temperature;
    if (method == "rk4")
    {
        k2 = (double *)malloc(sizeof(double) * Nx * Ny);
        k3 = (double *)malloc(sizeof(double) * Nx * Ny);
        k4 = (double *)malloc(sizeof(double) * Nx * Ny);
        inter_temperature = (double *)malloc(sizeof(double) * Nx * Ny);
    }

    // Fill initial temperature field and define heat source
    heat_transfer_field(k_field, grid, material, spatial_variation);
    temperature_field(temperature, grid, initial_temperature);
    heat_source_field(heat_source, grid, heat_generation);
    for (int i = 0; i < Nx * Ny; i++)
        solution[i] = temperature[i];

    // Time evolution
    std::cout << "Running CPU demo" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 1; t < timesteps; t++)
    {
        if (method == "euler")
        {
            // Evaluate function RHS
            evaluate_function_wrapper(k1, temperature, heat_source, k_field, fourier_grid_1, fourier_grid_2,
                                      real_grid_1, real_grid_2, forward_plan, backward_plan, grid, material, spatial_variation);

            // Integrate in time
            euler_update(temperature, k1, time_increment, grid);
        }
        else if (method == "rk4")
        {
            // Compute K1
            evaluate_function_wrapper(k1, temperature, heat_source, k_field, fourier_grid_1, fourier_grid_2,
                                      real_grid_1, real_grid_2, forward_plan, backward_plan, grid, material, spatial_variation);

            // Compute K2
            for (int i = 0; i < Nx * Ny; i++)
                inter_temperature[i] = temperature[i] + time_increment / 2 * k1[i];
            evaluate_function_wrapper(k2, inter_temperature, heat_source, k_field, fourier_grid_1, fourier_grid_2,
                                      real_grid_1, real_grid_2, forward_plan, backward_plan, grid, material, spatial_variation);

            // Compute K3
            for (int i = 0; i < Nx * Ny; i++)
                inter_temperature[i] = temperature[i] + time_increment / 2 * k2[i];
            evaluate_function_wrapper(k3, inter_temperature, heat_source, k_field, fourier_grid_1, fourier_grid_2,
                                      real_grid_1, real_grid_2, forward_plan, backward_plan, grid, material, spatial_variation);

            // Compute K4
            for (int i = 0; i < Nx * Ny; i++)
                inter_temperature[i] = temperature[i] + time_increment * k3[i];
            evaluate_function_wrapper(k4, inter_temperature, heat_source, k_field, fourier_grid_1, fourier_grid_2,
                                      real_grid_1, real_grid_2, forward_plan, backward_plan, grid, material, spatial_variation);

            // Integrate in time
            rk4_update(temperature, k1, k2, k3, k4, time_increment, grid);
        }
        else
        {
            std::cout << "Invalid method" << std::endl;
            return 1;
        }

        // Save solution
        for (int i = 0; i < Nx * Ny; i++)
            solution[t * Nx * Ny + i] = temperature[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    // Save solution to binary file
    std::string filename = "bin/solution_cpu.bin";
    write_solution(filename, solution, Nx, Ny, timesteps);

    // Free memory
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(real_grid_1);
    fftw_free(fourier_grid_1);
    free(k1);
    if (method == "rk4")
    {
        free(k2);
        free(k3);
        free(k4);
        free(inter_temperature);
    }
    free(temperature);
    free(heat_source);
    free(solution);
    std::cout << "Done!" << std::endl;
}
