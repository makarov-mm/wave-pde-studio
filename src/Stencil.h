#pragma once
#include <cstdint>
#include <cmath>

// The finite-difference stencil and the coloring, shared by both the CPU and
// CUDA back-ends so the two produce pixel-identical output. SC_HD expands to
// __host__ __device__ under nvcc and to nothing under a normal C++ compiler.
#ifdef __CUDACC__
    #define SC_HD __host__ __device__
#else
    #define SC_HD
#endif

// 5-point Laplacian with zero (Dirichlet) boundary: neighbors outside the grid
// count as 0, which makes the domain edges lightly absorbing.
SC_HD inline float sc_laplacian(const float* u, int x, int y, int w, int h) {
    float c  = u[y * w + x];
    float l  = (x > 0)     ? u[y * w + x - 1]     : 0.0f;
    float r  = (x < w - 1) ? u[y * w + x + 1]     : 0.0f;
    float d  = (y > 0)     ? u[(y - 1) * w + x]   : 0.0f;
    float up = (y < h - 1) ? u[(y + 1) * w + x]   : 0.0f;
    return l + r + d + up - 4.0f * c;
}

// One wave-equation step (leapfrog with damping) for a single cell.
SC_HD inline float sc_waveStep(const float* u, const float* uPrev,
                               int x, int y, int w, int h,
                               float k, float damping) {
    float lap  = sc_laplacian(u, x, y, w, h);
    float next = 2.0f * u[y * w + x] - uPrev[y * w + x] + k * lap;
    return next * (1.0f - damping);
}

// One heat-equation step (explicit) for a single cell.
SC_HD inline float sc_heatStep(const float* u, int x, int y, int w, int h,
                               float alpha) {
    float lap = sc_laplacian(u, x, y, w, h);
    return u[y * w + x] + alpha * lap;
}

SC_HD inline uint32_t sc_pack(float r, float g, float b) {
    int R = (int)(fminf(1.0f, fmaxf(0.0f, r)) * 255.0f + 0.5f);
    int G = (int)(fminf(1.0f, fmaxf(0.0f, g)) * 255.0f + 0.5f);
    int B = (int)(fminf(1.0f, fmaxf(0.0f, b)) * 255.0f + 0.5f);
    return 0xFF000000u | ((uint32_t)R << 16) | ((uint32_t)G << 8) | (uint32_t)B;
}

// Map a field value to a color. Wave fields are signed (diverging palette:
// crests and troughs in different colors); heat fields are non-negative
// (sequential palette).
SC_HD inline uint32_t sc_colorize(float v, int equation, int scheme) {
    if (equation == 0) {                             // wave: diverging
        float p = fmaxf(0.0f, fminf(1.0f,  v));
        float n = fmaxf(0.0f, fminf(1.0f, -v));
        if (scheme == 1)                             // ocean: cyan / orange
            return sc_pack(p + 0.1f * n, 0.5f * (p + n), n + 0.1f * p);
        if (scheme == 2) {                           // signed grayscale
            float s = 0.5f + 0.5f * fmaxf(-1.0f, fminf(1.0f, v));
            return sc_pack(s, s, s);
        }
        return sc_pack(p, 0.15f * (p + n), n);       // default: red(+)/blue(-)
    } else {                                         // heat: sequential
        float s = fmaxf(0.0f, fminf(1.0f, v));
        if (scheme == 1)                             // ocean
            return sc_pack(s * s, s, 0.4f + 0.6f * sqrtf(s));
        if (scheme == 2)                             // grayscale
            return sc_pack(s, s, s);
        return sc_pack(fminf(1.0f, s * 3.0f),        // thermal: black->red->yellow->white
                       fminf(1.0f, fmaxf(0.0f, s * 3.0f - 1.0f)),
                       fminf(1.0f, fmaxf(0.0f, s * 3.0f - 2.0f)));
    }
}
