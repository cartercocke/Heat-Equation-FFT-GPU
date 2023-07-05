#include <iostream>
#include <math.h>

#include "solution_io.h"

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

    // Read solutions from binary files
    float *cpu_solution = (float *)malloc(sizeof(float) * Nx * Ny * timesteps);
    float *gpu_solution = (float *)malloc(sizeof(float) * Nx * Ny * timesteps);

    std::string cpu_filename = "bin/solution_cpu.bin";
    std::string gpu_filename = "bin/solution_gpu.bin";

    read_solution(cpu_filename, cpu_solution, Nx, Ny, timesteps);
    read_solution(gpu_filename, gpu_solution, Nx, Ny, timesteps);

    std::cout << "Read solutions from binary files" << std::endl;

    // Compare solutions
    float max_diff = 1e-3;
    for (int i = 0; i < Nx * Ny * timesteps; i++)
    {
        if (abs(cpu_solution[i] - gpu_solution[i]) > max_diff)
        {
            std::cout << "Solutions differ at index " << i << std::endl;
            std::cout << "CPU: " << cpu_solution[i] << std::endl;
            std::cout << "GPU: " << gpu_solution[i] << std::endl;
            return 1;
        }
    }
    std::cout << "CPU and GPU solutions match!" << std::endl;

    // Clean up
    free(cpu_solution);
    free(gpu_solution);
}
