/**
 * @file cpu_heat_equation.cpp
 * @brief These are the functions used to solve the heat equation on the CPU
 *
 * @author Carter Cocke
 *
 * @date 06/03/2023
 */
#include <math.h>
#include <iostream>

#include <fftw3.h>
#include "helper_structs.h"

/**
 * @brief Compute the spatial frequency vector \f$k = (k_x, k_y)\f$
 *
 * @param i Grid index in x direction
 * @param j Grid index in y direction
 * @param grid Grid struct containing the grid size
 * @param kx Spatial frequency in x direction
 * @param ky Spatial frequency in y direction
 */
void spatial_frequency(const int i,
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
 * @brief Compute the normalized square of the norm of the spatial
 * frequency vector \f$\|k\|_2^2 / N\f$
 *
 * @param i Grid index in x direction
 * @param j Grid index in y direction
 * @param grid Grid struct containing the grid size
 * @return double squared norm of the spatial frequency vector
 */
double spatial_freq_square_norm(const int i,
                                const int j,
                                grid_size grid)
{
    double kx, ky;
    spatial_frequency(i, j, grid, &kx, &ky);
    return (kx * kx + ky * ky) / grid.Ntot;
}

/**
 * @brief Compute the Laplacian of the temperature field in Fourier space
 *
 * @param temperature_fourier complex field of the temperature in Fourier space
 * @param grid Grid struct containing the grid size
 */
void fourier_laplacian(fftw_complex *temperature_fourier,
                       grid_size grid)
{
    double k_norm2;
    for (int i = 0; i < grid.Nx; i++)
    {
        for (int j = 0; j < grid.Ny; j++)
        {
            k_norm2 = spatial_freq_square_norm(i, j, grid);
            temperature_fourier[i * grid.Ny + j][0] *= k_norm2;
            temperature_fourier[i * grid.Ny + j][1] *= k_norm2;
        }
    }
}

/**
 * @brief Compute the gradient of the temperature field in Fourier space
 *
 * @param fourier_grid_1 First fourier grid for x component of gradient
 * @param fourier_grid_2 Second fourier grid for y component of gradient
 * @param grid Grid struct containing the grid size
 */
void fourier_gradient(fftw_complex *fourier_grid_1,
                      fftw_complex *fourier_grid_2,
                      grid_size grid)
{
    double k1, k2;
    for (int i = 0; i < grid.Nx; i++)
    {
        for (int j = 0; j < grid.Ny; j++)
        {
            spatial_frequency(i, j, grid, &k1, &k2);
            fourier_grid_2[i * grid.Ny + j][0] = fourier_grid_1[i * grid.Ny + j][0] * k2 / grid.Ntot;
            fourier_grid_2[i * grid.Ny + j][1] = fourier_grid_1[i * grid.Ny + j][1] * k2 / grid.Ntot;
            fourier_grid_1[i * grid.Ny + j][0] *= k1 / grid.Ntot;
            fourier_grid_1[i * grid.Ny + j][1] *= k1 / grid.Ntot;
        }
    }
}

/**
 * @brief Compute the divergence of a vector field in Fourier space
 *
 * @param fourier_grid_1 FFT of first component of the vector field
 * @param fourier_grid_2 FFT of second component of the vector field
 * @param grid Grid struct containing the grid size
 */
void fourier_divergence(fftw_complex *fourier_grid_1,
                        fftw_complex *fourier_grid_2,
                        grid_size grid)
{
    double k1, k2;
    for (int i = 0; i < grid.Nx; i++)
    {
        for (int j = 0; j < grid.Ny; j++)
        {
            spatial_frequency(i, j, grid, &k1, &k2);
            fourier_grid_1[i * grid.Ny + j][0] *= k1 / grid.Ntot;
            fourier_grid_1[i * grid.Ny + j][1] *= k1 / grid.Ntot;
            fourier_grid_1[i * grid.Ny + j][0] += fourier_grid_2[i * grid.Ny + j][0] * k2 / grid.Ntot;
            fourier_grid_1[i * grid.Ny + j][1] += fourier_grid_2[i * grid.Ny + j][1] * k2 / grid.Ntot;
        }
    }
}

/**
 * @brief Compute the RHS of the heat equation with a given temperature field and heat source
 *
 * @param f_eval Evaluation of the function
 * @param temperature Temperature field to evaluate the function at
 * @param heat_source Heat source field to evaluate the function at
 * @param fourier_grid FFTW complex grid
 * @param real_grid FFTW real grid
 * @param forward_plan FFTW forward plan
 * @param backward_plan FFTW backward plan
 * @param grid Grid specifications
 * @param material Material specifications
 */
void evaluate_function(double *f_eval,
                       double *temperature,
                       double *heat_source,
                       fftw_complex *fourier_grid,
                       fftw_complex *real_grid,
                       fftw_plan forward_plan,
                       fftw_plan backward_plan,
                       grid_size grid,
                       material material)
{
    // Fill temperature FFT grid
    for (int i = 0; i < grid.Ntot; i++)
    {
        real_grid[i][0] = temperature[i];
        real_grid[i][1] = 0.0;
    }

    // FFT(temperature)
    fftw_execute_dft(forward_plan, real_grid, fourier_grid);

    // Compute laplacian(T_hat) in Fourier space
    fourier_laplacian(fourier_grid, grid);

    // IFFT(laplacian(T_hat))
    fftw_execute_dft(backward_plan, fourier_grid, real_grid);

    // Compute function evaluation q/rho*cp - alpha*laplacian(T)
    for (int i = 0; i < grid.Ntot; i++)
        f_eval[i] = (heat_source[i] - material.k * real_grid[i][0]) / (material.rho * material.cp);
}

/**
 * @brief Compute the RHS of the heat equation with a given temperature field and heat source
 *
 * @param f_eval Evaluation of the function
 * @param temperature Temperature field to evaluate the function at
 * @param heat_source Heat source field to evaluate the function at
 * @param fourier_grid_1 FFTW complex grid
 * @param fourier_grid_2 FFTW complex grid
 * @param real_grid_1 FFTW real grid
 * @param real_grid_2 FFTW real grid
 * @param forward_plan FFTW forward plan
 * @param backward_plan FFTW backward plan
 * @param grid Grid specifications
 * @param material Material specifications
 */
void evaluate_function_spatially_varying(double *f_eval,
                                         double *temperature,
                                         double *heat_source,
                                         double *k_field,
                                         fftw_complex *fourier_grid_1,
                                         fftw_complex *fourier_grid_2,
                                         fftw_complex *real_grid_1,
                                         fftw_complex *real_grid_2,
                                         fftw_plan forward_plan,
                                         fftw_plan backward_plan,
                                         grid_size grid,
                                         material material)
{
    // Fill temperature FFT grid
    for (int i = 0; i < grid.Ntot; i++)
    {
        real_grid_1[i][0] = temperature[i];
        real_grid_1[i][1] = 0.0;
    }

    // FFT(temperature)
    fftw_execute_dft(forward_plan, real_grid_1, fourier_grid_1);

    // Compute grad(T_hat) in Fourier space
    fourier_gradient(fourier_grid_1, fourier_grid_2, grid);

    // IFFT(grad(T_hat))
    fftw_execute_dft(backward_plan, fourier_grid_1, real_grid_1);
    fftw_execute_dft(backward_plan, fourier_grid_2, real_grid_2);

    // Compute k * grad(T)
    for (int i = 0; i < grid.Ntot; i++)
    {
        real_grid_1[i][0] *= k_field[i];
        real_grid_1[i][1] *= k_field[i];
        real_grid_2[i][0] *= k_field[i];
        real_grid_2[i][1] *= k_field[i];
    }

    // FFT(k * grad(T))
    fftw_execute_dft(forward_plan, real_grid_1, fourier_grid_1);
    fftw_execute_dft(forward_plan, real_grid_2, fourier_grid_2);

    // Compute div(k * grad(T)) in Fourier space
    fourier_divergence(fourier_grid_1, fourier_grid_2, grid);

    // IFFT(div(k * grad(T)))
    fftw_execute_dft(backward_plan, fourier_grid_1, real_grid_1);

    // Compute function evaluation q/rho*cp - div(k * grad(T))
    for (int i = 0; i < grid.Ntot; i++)
        f_eval[i] = (heat_source[i] - real_grid_1[i][0]) / (material.rho * material.cp);
}

/**
 * @brief Wrapper function for evaluating the RHS of the heat equation
 *
 * @copydoc evaluate_function_spatially_varying
 * @param spatial_variation Type of spatial variation to use
 */
void evaluate_function_wrapper(double *f_eval,
                               double *temperature,
                               double *heat_source,
                               double *k_field,
                               fftw_complex *fourier_grid_1,
                               fftw_complex *fourier_grid_2,
                               fftw_complex *real_grid_1,
                               fftw_complex *real_grid_2,
                               fftw_plan forward_plan,
                               fftw_plan backward_plan,
                               grid_size grid,
                               material material,
                               std::string spatial_variation)
{

    if (spatial_variation == "constant")
    {
        evaluate_function(f_eval, temperature, heat_source, fourier_grid_1, real_grid_1,
                          forward_plan, backward_plan, grid, material);
    }
    else if (spatial_variation == "half" || spatial_variation == "checkerboard")
    {
        evaluate_function_spatially_varying(f_eval, temperature, heat_source, k_field, fourier_grid_1, fourier_grid_2,
                                            real_grid_1, real_grid_2, forward_plan, backward_plan, grid, material);
    }
    else
    {
        std::cout << "Invalid spatial_variation defined" << std::endl;
        exit(1);
    }
}

/**
 * @brief Euler time step update
 *
 * @param temperature Temperature field at time t (updated to time t + dt)
 * @param f_eval Evaluation of the function at time t
 * @param dt Time increment dt
 * @param grid Grid struct containing the grid size
 */
void euler_update(double *temperature,
                  double *f_eval,
                  double dt,
                  grid_size grid)
{
    for (int i = 0; i < grid.Ntot; i++)
        temperature[i] += dt * f_eval[i];
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
void rk4_update(double *temperature,
                double *k1,
                double *k2,
                double *k3,
                double *k4,
                double dt,
                grid_size grid)
{
    for (int i = 0; i < grid.Ntot; i++)
        temperature[i] += dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}