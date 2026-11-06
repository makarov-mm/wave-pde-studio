#pragma once

// Plain types shared by the host code and the CUDA device code.
// Must stay free of Qt and of anything nvcc cannot compile.

enum class Equation : int {
    Wave,       // u_tt = c^2 * laplacian(u)      (leapfrog in time)
    Heat,       // u_t  = alpha * laplacian(u)    (explicit forward Euler)
    GrayScott   // reaction-diffusion of two coupled fields u ("food") and v ("consumer")
};

// How the 5-point stencil treats neighbors outside the grid.
enum class Boundary : int {
    Dirichlet,  // outside = 0      : waves reflect with inverted phase
    Neumann,    // outside = edge   : waves reflect like off a hard wall
    Periodic    // wrap-around      : the domain is a torus
};

enum class Scene : int { Empty, DoubleSlit, RingCavity, PillarLattice };

enum class Palette : int { Thermal, Ocean, Grayscale, Viridis, Magma };

struct SimParams {
    int      gridW    = 512;
    int      gridH    = 512;
    Equation equation = Equation::Wave;
    Boundary boundary = Boundary::Dirichlet;
    Scene    scene    = Scene::Empty;

    float waveSpeed = 0.5f;     // Courant number (0.05..0.7); k = waveSpeed^2
    float diffusion = 0.20f;    // heat alpha (0.01..0.24; the explicit scheme is stable below 0.25 in 2D)
    float damping   = 0.0005f;  // wave energy lost per step

    // Gray-Scott: du/dv are the diffusion rates for a 5-point Laplacian with dt = 1
    // (Pearson's classic scaling); feed/kill select the pattern regime.
    float feed = 0.0545f;
    float kill = 0.0620f;
    float du   = 0.2097f;
    float dv   = 0.1050f;

    // Interaction brush (left-drag pulses, right-drag walls).
    float brushRadius = 0.025f;  // fraction of the grid width
    float brushAmp    = 1.0f;

    Palette palette  = Palette::Thermal;
    int     substeps = 4;        // simulation steps advanced per rendered frame
};
