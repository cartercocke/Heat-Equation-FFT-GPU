/**
 * @file example_fields.cpp
 * @brief These functions define example fields for the heat equation
 *
 * @author Carter Cocke
 *
 * @date 06/03/2023
 */
#include <math.h>
#include <string>
#include <iostream>
// #include <algorithm>

#include "helper_structs.h"

/**
 * @brief Initial conditions with two sinusoidal waves of different frequencies
 *
 * @param temperature Temperature field
 * @param grid Grid struct containing the grid size
 * @param initial_temperature Type of initial temperature
 */
void temperature_field(double *temperature,
                       grid_size grid,
                       std::string initial_temperature)
{
    if (initial_temperature == "sine")
    {
        float x, y;
        for (int i = 0; i < grid.Nx; i++)
        {
            for (int j = 0; j < grid.Ny; j++)
            {
                x = M_PI * (double)i / grid.Nx;
                y = M_PI * (double)j / grid.Ny;
                temperature[i * grid.Ny + j] = 273 + 50 * sin(16 * x) * cos(16 * y) + 100 * sin(4 * x) * cos(4 * y);
            }
        }
    }
    else if (initial_temperature == "pyramid")
    {
        float x, y, xp, yp;
        for (int i = 0; i < grid.Nx; i++)
        {
            for (int j = 0; j < grid.Ny; j++)
            {
                x = (double)i / grid.Nx;
                y = (double)j / grid.Ny;
                xp = (double)(grid.Nx - i) / grid.Nx;
                yp = (double)(grid.Ny - j) / grid.Ny;
                temperature[i * grid.Ny + j] = 273 + (std::min(x, xp) + std::min(y, yp)) * 200;
            }
        }
    }
    else if (initial_temperature == "droplet")
    {
        float x, y, R;
        for (int i = 0; i < grid.Nx; i++)
        {
            for (int j = 0; j < grid.Ny; j++)
            {
                x = (double)i / grid.Nx;
                y = (double)j / grid.Ny;
                R = sqrt(pow(x - 0.5, 2) + pow(y - 0.5, 2));
                temperature[i * grid.Ny + j] = 273 + 75 * sin(10 * M_PI * R);
            }
        }
    }
    else if (initial_temperature == "constant")
        for (int i = 0; i < grid.Ntot; i++)
            temperature[i] = 273;
    else
    {
        std::cout << "Invalid initial temperature" << std::endl;
        exit(1);
    }
}

/**
 * @brief Heat source field with a single sinusoidal wave peaking in the middle
 *
 * @param heat_source Heat source field
 * @param grid Grid struct containing the grid size
 * @param heat_generation Type of heat generation
 */
void heat_source_field(double *heat_source,
                       grid_size grid,
                       std::string heat_generation)
{
    if (heat_generation == "sine")
    {
        float x, y;
        for (int i = 0; i < grid.Nx; i++)
        {
            for (int j = 0; j < grid.Ny; j++)
            {
                x = M_PI * (double)i / grid.Nx;
                y = M_PI * (double)j / grid.Ny;
                heat_source[i * grid.Ny + j] = 500 * sin(x) * sin(y);
            }
        }
    }
    else if (heat_generation == "constant")
        for (int i = 0; i < grid.Ntot; i++)
            heat_source[i] = 500;
    else if (heat_generation == "none")
        for (int i = 0; i < grid.Ntot; i++)
            heat_source[i] = 0;
    else
    {
        std::cout << "Invalid heat generation" << std::endl;
        exit(1);
    }
}

/**
 * @brief Build heat transfer coefficient field
 *
 * @param heat_source Heat source field
 * @param grid Grid struct containing the grid size
 * @param material Material struct containing the material properties
 * @param spatial_variation Type of spatial variation in k
 */
void heat_transfer_field(double *k_field,
                         grid_size grid,
                         material material,
                         std::string spatial_variation)
{
    if (spatial_variation == "constant")
        return;
    else if (spatial_variation == "half")
    {
        float x;
        for (int i = 0; i < grid.Nx; i++)
        {
            for (int j = 0; j < grid.Ny; j++)
            {
                x = (double)i / grid.Nx - 0.5;
                if (x < 0)
                    k_field[i * grid.Ny + j] = material.k;
                else
                    k_field[i * grid.Ny + j] = material.k / 10;
            }
        }
    }
    else if (spatial_variation == "checkerboard")
    {
        float x, y;
        for (int i = 0; i < grid.Nx; i++)
        {
            for (int j = 0; j < grid.Ny; j++)
            {
                x = (double)i / grid.Nx - 0.5;
                y = (double)j / grid.Ny - 0.5;
                if (x * y < 0)
                    k_field[i * grid.Ny + j] = material.k;
                else
                    k_field[i * grid.Ny + j] = material.k / 10;
            }
        }
    }
    else
    {
        std::cout << "Invalid spatial_variation" << std::endl;
        exit(1);
    }
}