#pragma once
#include <cstdint>

// Plain-old-data types shared between the Qt/host code and the CUDA device code.
// Must stay free of Qt includes so it can be compiled by nvcc.

enum class Equation : int {
    Wave      = 0,  // u_tt = c^2 * laplacian(u)      (leapfrog in time)
    Heat      = 1,  // u_t  = alpha * laplacian(u)    (explicit forward Euler)
    GrayScott = 2   // reaction-diffusion, two coupled fields u (feed) and v (kill)
};

// How the 5-point stencil treats neighbors outside the grid.
enum class BoundaryType : int {
    Dirichlet = 0,  // outside = 0        : edges reflect with phase inversion
    Neumann   = 1,  // outside = edge     : edges reflect like hard walls
    Periodic  = 2   // wrap-around        : the domain is a torus
};

struct SimParams {
    int      gridW    = 512;
    int      gridH    = 512;
    Equation equation = Equation::Wave;
    int      boundary = 0;   // BoundaryType as int (kept POD-simple for nvcc)
    int      scene    = 0;   // 0 empty, 1 double slit, 2 ring cavity, 3 pillar lattice

    float    waveSpeed = 0.5f;    // Courant number (0.05..0.7); k = waveSpeed^2
    float    diffusion = 0.20f;   // heat alpha (0.01..0.24, stable < 0.25 in 2D)
    float    damping   = 0.0005f; // wave energy decay per step

    // Gray-Scott reaction-diffusion. du/dv are the diffusion rates matched to a
    // 5-point Laplacian with dt = 1 (Pearson's classic scaling); feed/kill select
    // the pattern regime (coral, mitosis, worms, ...).
    float    feed = 0.0545f;
    float    kill = 0.0620f;
    float    du   = 0.2097f;
    float    dv   = 0.1050f;

    // Interaction brush (left-drag pulses, right-drag walls).
    float    brushRadius = 0.025f; // as a fraction of the grid width
    float    brushAmp    = 1.0f;

    int      colorScheme = 0;     // 0 thermal/diverging, 1 ocean, 2 gray, 3 viridis, 4 magma
    int      substeps    = 4;     // simulation steps advanced per rendered frame
};
