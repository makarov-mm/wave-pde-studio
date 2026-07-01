#pragma once
#include <cstdint>

// Plain-old-data types shared between the Qt/host code and the CUDA device code.
// Must stay free of Qt includes so it can be compiled by nvcc.

enum class Equation : int {
    Wave = 0,   // u_tt = c^2 * laplacian(u)   (finite-difference leapfrog in time)
    Heat = 1    // u_t  = alpha * laplacian(u) (explicit forward Euler in time)
};

struct SimParams {
    int      gridW     = 512;
    int      gridH     = 512;
    Equation equation  = Equation::Wave;

    float    waveSpeed = 0.5f;    // Courant number (0.05..0.7); k = waveSpeed^2
    float    diffusion = 0.20f;   // heat alpha (0.01..0.24, stable < 0.25 in 2D)
    float    damping   = 0.0005f; // wave energy decay per step

    int      colorScheme = 0;     // 0 = thermal/diverging, 1 = ocean, 2 = grayscale
    int      substeps    = 4;     // simulation steps advanced per rendered frame
};
