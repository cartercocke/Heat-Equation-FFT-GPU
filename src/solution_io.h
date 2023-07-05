#pragma once

#include <stdio.h>
#include <iostream>

#include "helper_structs.h"

void write_solution(std::string filename,
                    float *solution,
                    int Nx,
                    int Ny,
                    int timesteps);

void read_solution(std::string filename,
                   float *solution,
                   int Nx,
                   int Ny,
                   int timesteps);

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
                material *material);