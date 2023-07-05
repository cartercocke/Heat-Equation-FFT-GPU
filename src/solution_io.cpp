/**
 * @file solution_io.cpp
 * @brief These are the helper functions used for I/O
 *
 * @author Carter Cocke
 *
 * @date 06/03/2023
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "helper_structs.h"

/**
 * @brief Write a solution to a binary file
 *
 * @param filename Name of the file
 * @param solution solution array
 * @param Nx Grid size in x direction
 * @param Ny Grid size in y direction
 * @param timesteps Number of timesteps
 */
void write_solution(std::string filename,
                    float *solution,
                    int Nx,
                    int Ny,
                    int timesteps)
{
    FILE *file;
    file = fopen(filename.c_str(), "wb");
    if (file == NULL)
    {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }
    fwrite(solution, sizeof(float), Nx * Ny * timesteps, file);
    fclose(file);
}

/**
 * @brief Read a solution to a binary file
 *
 * @param filename Name of the file
 * @param solution solution array
 * @param Nx Grid size in x direction
 * @param Ny Grid size in y direction
 * @param timesteps Number of timesteps
 */
void read_solution(std::string filename,
                   float *solution,
                   int Nx,
                   int Ny,
                   int timesteps)
{
    FILE *file;
    file = fopen(filename.c_str(), "rb");
    if (file == NULL)
    {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }
    fread(solution, sizeof(float), Nx * Ny * timesteps, file);
    fclose(file);
}

/**
 * @brief Read input parameters from a text file
 *
 * @param filename Input file name
 * @param Nx Grid size in x direction
 * @param Ny Grid size in y direction
 * @param timesteps Number of timesteps
 * @param time_increment Time increment
 * @param method Integration method to use
 * @param spatial_variation Spatial variation of heat transfer coefficient k
 * @param grid grid_size struct
 * @param material material struct
 */
void read_input(std::string filename,
                int *Nx,
                int *Ny,
                int *timesteps,
                double *time_increment,
                std::string *method,
                std::string *spatial_variation,
                std::string *initial_temperature,
                std::string *heat_generation,
                grid_size *grid,
                material *material)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string key;
        if (!(iss >> key))
        {
            continue;
        }
        if (key[0] == '#')
        {
            continue;
        }
        else if (key == "Nx")
        {
            iss >> *Nx;
        }
        else if (key == "Ny")
        {
            iss >> *Ny;
        }
        else if (key == "timesteps")
        {
            iss >> *timesteps;
        }
        else if (key == "time_increment")
        {
            iss >> *time_increment;
        }
        else if (key == "method")
        {
            iss >> *method;
        }
        else if (key == "spatial_variation")
        {
            iss >> *spatial_variation;
        }
        else if (key == "initial_temperature")
        {
            iss >> *initial_temperature;
        }
        else if (key == "heat_generation")
        {
            iss >> *heat_generation;
        }
        else if (key == "k")
        {
            iss >> material->k;
        }
        else if (key == "rho")
        {
            iss >> material->rho;
        }
        else if (key == "cp")
        {
            iss >> material->cp;
        }
        else
        {
            std::cout << "Unknown key " << key << std::endl;
        }
    }
    if (*Nx == 0 || *Ny == 0 || *timesteps == 0 || *time_increment == 0 || *method == "" || *spatial_variation == "" || *initial_temperature == "" || *heat_generation == "" || material->k == 0 || material->rho == 0 || material->cp == 0)
    {
        std::cout << "Missing or invalid input parameter" << std::endl;
        exit(1);
    }
    grid->Nx = *Nx;
    grid->Ny = *Ny;
    grid->Ntot = *Nx * *Ny;
    infile.close();
}