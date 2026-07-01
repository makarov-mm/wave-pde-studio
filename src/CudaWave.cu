// CudaWave.cu
// GPU back-end: one thread per grid cell advances the finite-difference stencil.
// Uses the same stencil math as the CPU path (Stencil.h) so results match.

#include "CudaWave.h"
#include "Stencil.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// Persistent device buffers, grown on demand.
static float* g_a = nullptr;   // u
static float* g_b = nullptr;   // uPrev
static float* g_c = nullptr;   // next
static int    g_cap = 0;

__global__ void waveKernel(float* next, const float* u, const float* uPrev,
                           int w, int h, float k, float damping) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    next[y * w + x] = sc_waveStep(u, uPrev, x, y, w, h, k, damping);
}

__global__ void heatKernel(float* next, const float* u,
                           int w, int h, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    next[y * w + x] = sc_heatStep(u, x, y, w, h, alpha);
}

static void ensureCapacity(int count) {
    if (count <= g_cap) return;
    if (g_a) cudaFree(g_a);
    if (g_b) cudaFree(g_b);
    if (g_c) cudaFree(g_c);
    size_t bytes = (size_t)count * sizeof(float);
    cudaMalloc(&g_a, bytes);
    cudaMalloc(&g_b, bytes);
    cudaMalloc(&g_c, bytes);
    g_cap = count;
}

bool cudaAvailable() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

void cudaStepField(const SimParams& p, float* u, float* uPrev,
                   int substeps, double* elapsedMs) {
    int    count = p.gridW * p.gridH;
    size_t bytes = (size_t)count * sizeof(float);
    ensureCapacity(count);

    dim3 block(16, 16);
    dim3 grid((p.gridW + block.x - 1) / block.x,
              (p.gridH + block.y - 1) / block.y);

    float k     = p.waveSpeed * p.waveSpeed;
    float alpha = p.diffusion;
    bool  wave  = (p.equation == Equation::Wave);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Upload current state, iterate on the device (rotating pointers, no copies
    // between steps), then download the result. The reported time is the honest
    // "wall time to advance N steps and get the field back".
    float* d_u    = g_a;
    float* d_prev = g_b;
    float* d_next = g_c;
    cudaMemcpy(d_u,    u,     bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev, uPrev, bytes, cudaMemcpyHostToDevice);

    for (int s = 0; s < substeps; ++s) {
        if (wave) {
            waveKernel<<<grid, block>>>(d_next, d_u, d_prev, p.gridW, p.gridH, k, p.damping);
            // rotate: prev <- u, u <- next
            float* old = d_prev;
            d_prev = d_u;
            d_u    = d_next;
            d_next = old;
        } else {
            heatKernel<<<grid, block>>>(d_next, d_u, p.gridW, p.gridH, alpha);
            std::swap(d_u, d_next);
        }
    }

    cudaMemcpy(u,     d_u,    bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(uPrev, d_prev, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *elapsedMs = (double)ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void cudaShutdown() {
    if (g_a) cudaFree(g_a);
    if (g_b) cudaFree(g_b);
    if (g_c) cudaFree(g_c);
    g_a = g_b = g_c = nullptr;
    g_cap = 0;
}
