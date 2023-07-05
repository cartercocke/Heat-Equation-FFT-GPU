/**
* @file helper_structs.h
* @brief This file contains the helper structs for the project
*
* @author Carter Cocke
*
* @date 06/03/2023
*/
#pragma once

/**
 * @brief Grid dimensions
 *
 * @param Nx Number of grid points in the x direction
 * @param Ny Number of grid points in the y direction
 * @param Ntot Total number of grid points
 */
struct
{
    int Nx; //!< Number of grid points in the x direction
    int Ny; //!< Number of grid points in the y direction
    int Ntot; //!< Total number of grid points
} typedef grid_size;

/**
 * @brief Material properties
 *
 * @param cp Specific heat capacity
 * @param rho Density
 * @param k Thermal conductivity
 */
struct
{
    double cp; //!< Specific heat capacity
    double rho; //!< Density
    double k; //!< Thermal conductivity
} typedef material;