#pragma once
#include "Types.h"

// The finite-difference math shared by every back-end: the CPU loops and both
// CUDA kernel families call these exact functions, so all of them evaluate the
// same operations in the same order and produce bit-identical fields (the build
// disables FMA contraction on CPU and GPU to keep it that way).
//
// SC_HD expands to __host__ __device__ under nvcc and to nothing otherwise.
#ifdef __CUDACC__
    #define SC_HD __host__ __device__
#else
    #define SC_HD
#endif

namespace stencil {

// Values that wall cells are pinned to every step.
inline constexpr float kWallWave = 0.0f;   // wave: the field is held at rest
inline constexpr float kWallHeat = 0.0f;   // heat: walls act as a cold sink
inline constexpr float kWallU    = 1.0f;   // Gray-Scott: inert medium (u at rest,
inline constexpr float kWallV    = 0.0f;   //             no consumer)

// Updated values smaller than this are stored as zero. They are far below anything
// visible, and dropping them keeps every intermediate result out of the subnormal
// range: decaying waves and Gray-Scott's fading v produce many subnormals, which
// make CPU arithmetic several times slower (GPUs handle them at full speed).
inline constexpr float kFlushBelow = 1e-12f;

SC_HD inline float flushTiny(float x) {
    return (x > -kFlushBelow && x < kFlushBelow) ? 0.0f : x;
}

// u[x, y], applying the boundary rule when (x, y) lies outside the grid.
SC_HD inline float sample(const float* u, int x, int y, int w, int h, Boundary boundary) {
    if (static_cast<unsigned>(x) < static_cast<unsigned>(w) &&
        static_cast<unsigned>(y) < static_cast<unsigned>(h))
        return u[y * w + x];

    switch (boundary) {
    case Boundary::Neumann:     // clamp to the edge: zero normal derivative
        x = (x < 0) ? 0 : ((x >= w) ? w - 1 : x);
        y = (y < 0) ? 0 : ((y >= h) ? h - 1 : y);
        return u[y * w + x];
    case Boundary::Periodic:    // wrap around
        x %= w; if (x < 0) x += w;
        y %= h; if (y < 0) y += h;
        return u[y * w + x];
    case Boundary::Dirichlet:
    default:
        return 0.0f;
    }
}

// The 5-point Laplacian. `top` is row y - 1, `bottom` is row y + 1.
SC_HD inline float laplacian5(float center, float left, float right, float top, float bottom) {
    return left + right + top + bottom - 4.0f * center;
}

// Laplacian at any cell, honoring the boundary rule. Interior obstacles need no
// special case: wall cells hold their pinned value, which is what neighbors read.
SC_HD inline float laplacian(const float* u, int x, int y, int w, int h, Boundary boundary) {
    return laplacian5(u[y * w + x],
                      sample(u, x - 1, y, w, h, boundary),
                      sample(u, x + 1, y, w, h, boundary),
                      sample(u, x, y - 1, w, h, boundary),
                      sample(u, x, y + 1, w, h, boundary));
}

// Laplacian at an interior cell (1 <= x < w - 1, 1 <= y < h - 1) with flat index i:
// all neighbors are inside the grid, so no boundary handling is needed.
SC_HD inline float laplacianInterior(const float* u, int i, int w) {
    return laplacian5(u[i], u[i - 1], u[i + 1], u[i - w], u[i + w]);
}

// Wave equation: leapfrog step with damping.
SC_HD inline float waveUpdate(float u, float uPrev, float lap, float k, float damping) {
    return flushTiny((2.0f * u - uPrev + k * lap) * (1.0f - damping));
}

// Heat equation: explicit forward Euler step.
SC_HD inline float heatUpdate(float u, float lap, float alpha) {
    return flushTiny(u + alpha * lap);
}

struct GrayScottCell {
    float u;
    float v;
};

// Gray-Scott reaction-diffusion step for one cell:
//   u' = u + du*lap(u) - u*v^2 + F*(1 - u)
//   v' = v + dv*lap(v) + u*v^2 - (F + K)*v
SC_HD inline GrayScottCell grayScottUpdate(float u, float v, float lapU, float lapV,
                                           float du, float dv, float feed, float kill) {
    const float uvv = u * v * v;
    return { flushTiny(u + du * lapU - uvv + feed * (1.0f - u)),
             flushTiny(v + dv * lapV + uvv - (feed + kill) * v) };
}

} // namespace stencil
