#include <math.h>

#include <fftw3.h>
#include "helper_structs.h"

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
                               std::string spatial_variation);

void euler_update(double *temperature,
                  double *f_eval,
                  double dt,
                  grid_size grid);

void rk4_update(double *temperature,
                double *k1,
                double *k2,
                double *k3,
                double *k4,
                double dt,
                grid_size grid);