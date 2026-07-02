#pragma once
#include <cstdint>
#include <cmath>

// The finite-difference stencils and the coloring, shared by the CPU and CUDA
// back-ends so all of them produce pixel-identical output. SC_HD expands to
// __host__ __device__ under nvcc and to nothing under a normal C++ compiler.
#ifdef __CUDACC__
    #define SC_HD __host__ __device__
#else
    #define SC_HD
#endif

// Fetch u[x, y] applying the selected boundary rule when (x, y) is outside the
// grid. 0 = Dirichlet (outside is 0), 1 = Neumann (clamp to the edge value,
// reflective), 2 = periodic (wrap around).
SC_HD inline float sc_at(const float* u, int x, int y, int w, int h, int boundary) {
    if ((unsigned)x < (unsigned)w && (unsigned)y < (unsigned)h)
        return u[y * w + x];
    if (boundary == 1) {                       // Neumann: clamp
        x = (x < 0) ? 0 : ((x >= w) ? w - 1 : x);
        y = (y < 0) ? 0 : ((y >= h) ? h - 1 : y);
        return u[y * w + x];
    }
    if (boundary == 2) {                       // periodic: wrap
        x %= w; if (x < 0) x += w;
        y %= h; if (y < 0) y += h;
        return u[y * w + x];
    }
    return 0.0f;                               // Dirichlet
}

// 5-point Laplacian honoring the boundary rule. Interior walls are handled one
// level up: wall cells are pinned to a fixed value each step, so their stored
// value is what neighbors read (Dirichlet-style internal obstacles).
SC_HD inline float sc_laplacian(const float* u, int x, int y, int w, int h, int boundary) {
    float c  = u[y * w + x];
    float l  = sc_at(u, x - 1, y, w, h, boundary);
    float r  = sc_at(u, x + 1, y, w, h, boundary);
    float d  = sc_at(u, x, y - 1, w, h, boundary);
    float up = sc_at(u, x, y + 1, w, h, boundary);
    return l + r + d + up - 4.0f * c;
}

// One wave-equation step (leapfrog with damping) for a single cell.
SC_HD inline float sc_waveStep(const float* u, const float* uPrev, const uint8_t* mask,
                               int x, int y, int w, int h,
                               float k, float damping, int boundary) {
    int idx = y * w + x;
    if (mask[idx]) return 0.0f;                          // wall: field pinned to 0
    float lap  = sc_laplacian(u, x, y, w, h, boundary);
    float next = 2.0f * u[idx] - uPrev[idx] + k * lap;
    return next * (1.0f - damping);
}

// One heat-equation step (explicit) for a single cell.
SC_HD inline float sc_heatStep(const float* u, const uint8_t* mask,
                               int x, int y, int w, int h,
                               float alpha, int boundary) {
    int idx = y * w + x;
    if (mask[idx]) return 0.0f;                          // wall acts as a cold sink
    float lap = sc_laplacian(u, x, y, w, h, boundary);
    return u[idx] + alpha * lap;
}

// One Gray-Scott reaction-diffusion step for a single cell (updates both fields).
// u is the "food" chemical (rest state 1), v the "consumer" (rest state 0):
//   u' = u + du*lap(u) - u*v^2 + F*(1 - u)
//   v' = v + dv*lap(v) + u*v^2 - (F + K)*v
SC_HD inline void sc_gsStep(const float* u, const float* v, const uint8_t* mask,
                            int x, int y, int w, int h,
                            float du, float dv, float F, float K, int boundary,
                            float* outU, float* outV) {
    int idx = y * w + x;
    if (mask[idx]) { *outU = 1.0f; *outV = 0.0f; return; }   // wall: inert medium
    float uu   = u[idx];
    float vv   = v[idx];
    float uvv  = uu * vv * vv;
    float lapU = sc_laplacian(u, x, y, w, h, boundary);
    float lapV = sc_laplacian(v, x, y, w, h, boundary);
    *outU = uu + du * lapU - uvv + F * (1.0f - uu);
    *outV = vv + dv * lapV + uvv - (F + K) * vv;
}

SC_HD inline uint32_t sc_pack(float r, float g, float b) {
    int R = (int)(fminf(1.0f, fmaxf(0.0f, r)) * 255.0f + 0.5f);
    int G = (int)(fminf(1.0f, fmaxf(0.0f, g)) * 255.0f + 0.5f);
    int B = (int)(fminf(1.0f, fmaxf(0.0f, b)) * 255.0f + 0.5f);
    return 0xFF000000u | ((uint32_t)R << 16) | ((uint32_t)G << 8) | (uint32_t)B;
}

// Piecewise-linear color lookup over n control points given in 0..255.
SC_HD inline uint32_t sc_lut(float t, const float (*P)[3], int n) {
    t = fminf(1.0f, fmaxf(0.0f, t));
    float f = t * (float)(n - 1);
    int   i = (int)f;
    if (i > n - 2) i = n - 2;
    float s = f - (float)i;
    float r = P[i][0] + (P[i + 1][0] - P[i][0]) * s;
    float g = P[i][1] + (P[i + 1][1] - P[i][1]) * s;
    float b = P[i][2] + (P[i + 1][2] - P[i][2]) * s;
    return sc_pack(r / 255.0f, g / 255.0f, b / 255.0f);
}

SC_HD inline uint32_t sc_viridis(float t) {
    const float P[9][3] = { { 68,   1,  84}, { 72,  40, 120}, { 62,  74, 137},
                            { 49, 104, 142}, { 38, 130, 142}, { 31, 158, 137},
                            { 53, 183, 121}, {109, 205,  89}, {253, 231,  37} };
    return sc_lut(t, P, 9);
}

SC_HD inline uint32_t sc_magma(float t) {
    const float P[9][3] = { {  0,   0,   4}, { 28,  16,  68}, { 79,  18, 123},
                            {129,  37, 129}, {181,  54, 122}, {229,  80, 100},
                            {251, 135,  97}, {254, 194, 135}, {252, 253, 191} };
    return sc_lut(t, P, 9);
}

// Map a field value to a color. `diverging` != 0 means the field is signed
// (wave crests/troughs); otherwise the field is non-negative (heat, Gray-Scott).
SC_HD inline uint32_t sc_colorize(float v, int diverging, int scheme) {
    if (scheme == 3 || scheme == 4) {                    // viridis / magma
        float t = diverging ? 0.5f + 0.5f * fmaxf(-1.0f, fminf(1.0f, v))
                            : fmaxf(0.0f, fminf(1.0f, v));
        return (scheme == 3) ? sc_viridis(t) : sc_magma(t);
    }
    if (diverging) {
        float p = fmaxf(0.0f, fminf(1.0f,  v));
        float n = fmaxf(0.0f, fminf(1.0f, -v));
        if (scheme == 1)                                 // ocean: cyan / orange
            return sc_pack(p + 0.1f * n, 0.5f * (p + n), n + 0.1f * p);
        if (scheme == 2) {                               // signed grayscale
            float s = 0.5f + 0.5f * fmaxf(-1.0f, fminf(1.0f, v));
            return sc_pack(s, s, s);
        }
        return sc_pack(p, 0.15f * (p + n), n);           // default: red(+)/blue(-)
    } else {
        float s = fmaxf(0.0f, fminf(1.0f, v));
        if (scheme == 1)                                 // ocean
            return sc_pack(s * s, s, 0.4f + 0.6f * sqrtf(s));
        if (scheme == 2)                                 // grayscale
            return sc_pack(s, s, s);
        return sc_pack(fminf(1.0f, s * 3.0f),            // thermal: black->red->yellow->white
                       fminf(1.0f, fmaxf(0.0f, s * 3.0f - 1.0f)),
                       fminf(1.0f, fmaxf(0.0f, s * 3.0f - 2.0f)));
    }
}
