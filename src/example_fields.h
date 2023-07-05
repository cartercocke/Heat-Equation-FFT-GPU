#pragma once

#include "helper_structs.h"

void temperature_field(double *temperature,
                       grid_size grid,
                       std::string initial_temperature);

void heat_source_field(double *heat_source,
                       grid_size grid,
                       std::string heat_generation);

void heat_transfer_field(double *heat_source,
                         grid_size grid,
                         material material,
                         std::string spatial_variation);