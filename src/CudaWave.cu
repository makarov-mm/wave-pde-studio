// CudaWave.cu
// GPU back-end: one thread per grid cell advances the finite-difference stencil.
// Uses the same stencil math as the CPU path (Stencil.h) so results match.
//
// Two kernel families are provided per equation:
//   * naive  — every thread reads its 5 stencil values straight from global
//              memory (the L2/L1 caches absorb most of the redundancy);
//   * tiled  — each block first stages a (TILE+2)^2 tile of the field, including
//              a one-cell halo, into shared memory, then computes the stencil
//              from shared memory. This is the classic stencil optimization;
//              the app benchmarks both so the difference can be measured.

#include "CudaWave.h"
#include "Stencil.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#define TILE 16

// Persistent device buffers, grown on demand.
static float*   g_a = nullptr;      // u
static float*   g_b = nullptr;      // uPrev (wave)
static float*   g_c = nullptr;      // next
static float*   g_v = nullptr;      // v      (Gray-Scott)
static float*   g_v2 = nullptr;     // vNext  (Gray-Scott)
static uint8_t* g_mask = nullptr;
static int      g_cap = 0;
static int      g_maskVersion = -1; // last uploaded mask version

static void ckCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------
// Naive kernels: straight application of the shared stencil functions.
// ---------------------------------------------------------------------------

__global__ void waveKernel(float* next, const float* u, const float* uPrev,
                           const uint8_t* mask, int w, int h,
                           float k, float damping, int boundary) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    next[y * w + x] = sc_waveStep(u, uPrev, mask, x, y, w, h, k, damping, boundary);
}

__global__ void heatKernel(float* next, const float* u, const uint8_t* mask,
                           int w, int h, float alpha, int boundary) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    next[y * w + x] = sc_heatStep(u, mask, x, y, w, h, alpha, boundary);
}

__global__ void gsKernel(float* nextU, float* nextV,
                         const float* u, const float* v, const uint8_t* mask,
                         int w, int h, float du, float dv,
                         float F, float K, int boundary) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    sc_gsStep(u, v, mask, x, y, w, h, du, dv, F, K, boundary,
              &nextU[idx], &nextV[idx]);
}

// ---------------------------------------------------------------------------
// Tiled kernels: stage the tile + halo in shared memory, then apply the same
// arithmetic (same operand order, so the result stays bit-identical).
// ---------------------------------------------------------------------------

// Loads the (TILE+2)^2 tile for field `f` around this block into `t`.
// Every thread loads its center cell; edge threads additionally load the halo.
__device__ inline void loadTile(float t[TILE + 2][TILE + 2], const float* f,
                                int x, int y, int w, int h, int boundary) {
    int tx = threadIdx.x, ty = threadIdx.y;
    t[ty + 1][tx + 1] = sc_at(f, x, y, w, h, boundary);
    if (tx == 0)        t[ty + 1][0]        = sc_at(f, x - 1, y, w, h, boundary);
    if (tx == TILE - 1) t[ty + 1][TILE + 1] = sc_at(f, x + 1, y, w, h, boundary);
    if (ty == 0)        t[0][tx + 1]        = sc_at(f, x, y - 1, w, h, boundary);
    if (ty == TILE - 1) t[TILE + 1][tx + 1] = sc_at(f, x, y + 1, w, h, boundary);
}

__device__ inline float tileLaplacian(const float t[TILE + 2][TILE + 2]) {
    int tx = threadIdx.x, ty = threadIdx.y;
    float c  = t[ty + 1][tx + 1];
    float l  = t[ty + 1][tx];
    float r  = t[ty + 1][tx + 2];
    float d  = t[ty][tx + 1];
    float up = t[ty + 2][tx + 1];
    return l + r + d + up - 4.0f * c;
}

__global__ void waveKernelTiled(float* next, const float* u, const float* uPrev,
                                const uint8_t* mask, int w, int h,
                                float k, float damping, int boundary) {
    __shared__ float t[TILE + 2][TILE + 2];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    loadTile(t, u, x, y, w, h, boundary);
    __syncthreads();
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    if (mask[idx]) { next[idx] = 0.0f; return; }
    float lap = tileLaplacian(t);
    float n   = 2.0f * t[threadIdx.y + 1][threadIdx.x + 1] - uPrev[idx] + k * lap;
    next[idx] = n * (1.0f - damping);
}

__global__ void heatKernelTiled(float* next, const float* u, const uint8_t* mask,
                                int w, int h, float alpha, int boundary) {
    __shared__ float t[TILE + 2][TILE + 2];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    loadTile(t, u, x, y, w, h, boundary);
    __syncthreads();
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    if (mask[idx]) { next[idx] = 0.0f; return; }
    next[idx] = t[threadIdx.y + 1][threadIdx.x + 1] + alpha * tileLaplacian(t);
}

__global__ void gsKernelTiled(float* nextU, float* nextV,
                              const float* u, const float* v, const uint8_t* mask,
                              int w, int h, float du, float dv,
                              float F, float K, int boundary) {
    __shared__ float tu[TILE + 2][TILE + 2];
    __shared__ float tv[TILE + 2][TILE + 2];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    loadTile(tu, u, x, y, w, h, boundary);
    loadTile(tv, v, x, y, w, h, boundary);
    __syncthreads();
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    if (mask[idx]) { nextU[idx] = 1.0f; nextV[idx] = 0.0f; return; }
    float uu  = tu[threadIdx.y + 1][threadIdx.x + 1];
    float vv  = tv[threadIdx.y + 1][threadIdx.x + 1];
    float uvv = uu * vv * vv;
    nextU[idx] = uu + du * tileLaplacian(tu) - uvv + F * (1.0f - uu);
    nextV[idx] = vv + dv * tileLaplacian(tv) + uvv - (F + K) * vv;
}

// ---------------------------------------------------------------------------

static void ensureCapacity(int count) {
    if (count <= g_cap) return;
    if (g_a)    cudaFree(g_a);
    if (g_b)    cudaFree(g_b);
    if (g_c)    cudaFree(g_c);
    if (g_v)    cudaFree(g_v);
    if (g_v2)   cudaFree(g_v2);
    if (g_mask) cudaFree(g_mask);
    size_t bytes = (size_t)count * sizeof(float);
    ckCuda(cudaMalloc(&g_a,  bytes),  "malloc u");
    ckCuda(cudaMalloc(&g_b,  bytes),  "malloc uPrev");
    ckCuda(cudaMalloc(&g_c,  bytes),  "malloc next");
    ckCuda(cudaMalloc(&g_v,  bytes),  "malloc v");
    ckCuda(cudaMalloc(&g_v2, bytes),  "malloc vNext");
    ckCuda(cudaMalloc(&g_mask, (size_t)count), "malloc mask");
    g_cap = count;
    g_maskVersion = -1;   // buffers are new: force a mask re-upload
}

bool cudaAvailable() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

void cudaStepField(const SimParams& p, float* u, float* uPrev, float* v,
                   const uint8_t* mask, int maskVersion, int substeps, bool tiled) {
    int    count = p.gridW * p.gridH;
    size_t bytes = (size_t)count * sizeof(float);
    ensureCapacity(count);

    dim3 block(TILE, TILE);
    dim3 grid((p.gridW + TILE - 1) / TILE,
              (p.gridH + TILE - 1) / TILE);

    float k = p.waveSpeed * p.waveSpeed;

    // The obstacle mask changes rarely (scene switches, wall painting), so it is
    // uploaded only when its version differs from what the device already has.
    if (maskVersion != g_maskVersion) {
        ckCuda(cudaMemcpy(g_mask, mask, (size_t)count, cudaMemcpyHostToDevice),
               "upload mask");
        g_maskVersion = maskVersion;
    }

    // Upload current state, iterate on the device (rotating pointers, no copies
    // between steps), then download the result. The caller times the whole call
    // with a wall clock, so the reported cost honestly includes the transfers.
    float* d_u    = g_a;
    float* d_prev = g_b;
    float* d_next = g_c;
    float* d_v    = g_v;
    float* d_vn   = g_v2;

    ckCuda(cudaMemcpy(d_u, u, bytes, cudaMemcpyHostToDevice), "upload u");
    if (p.equation == Equation::Wave)
        ckCuda(cudaMemcpy(d_prev, uPrev, bytes, cudaMemcpyHostToDevice), "upload uPrev");
    if (p.equation == Equation::GrayScott)
        ckCuda(cudaMemcpy(d_v, v, bytes, cudaMemcpyHostToDevice), "upload v");

    for (int s = 0; s < substeps; ++s) {
        switch (p.equation) {
        case Equation::Wave: {
            if (tiled)
                waveKernelTiled<<<grid, block>>>(d_next, d_u, d_prev, g_mask,
                                                 p.gridW, p.gridH, k, p.damping, p.boundary);
            else
                waveKernel<<<grid, block>>>(d_next, d_u, d_prev, g_mask,
                                            p.gridW, p.gridH, k, p.damping, p.boundary);
            float* old = d_prev;    // rotate: prev <- u, u <- next
            d_prev = d_u;
            d_u    = d_next;
            d_next = old;
            break;
        }
        case Equation::Heat: {
            if (tiled)
                heatKernelTiled<<<grid, block>>>(d_next, d_u, g_mask,
                                                 p.gridW, p.gridH, p.diffusion, p.boundary);
            else
                heatKernel<<<grid, block>>>(d_next, d_u, g_mask,
                                            p.gridW, p.gridH, p.diffusion, p.boundary);
            std::swap(d_u, d_next);
            break;
        }
        case Equation::GrayScott: {
            if (tiled)
                gsKernelTiled<<<grid, block>>>(d_next, d_vn, d_u, d_v, g_mask,
                                               p.gridW, p.gridH, p.du, p.dv,
                                               p.feed, p.kill, p.boundary);
            else
                gsKernel<<<grid, block>>>(d_next, d_vn, d_u, d_v, g_mask,
                                          p.gridW, p.gridH, p.du, p.dv,
                                          p.feed, p.kill, p.boundary);
            std::swap(d_u, d_next);
            std::swap(d_v, d_vn);
            break;
        }
        }
    }

    ckCuda(cudaMemcpy(u, d_u, bytes, cudaMemcpyDeviceToHost), "download u");
    if (p.equation == Equation::Wave)
        ckCuda(cudaMemcpy(uPrev, d_prev, bytes, cudaMemcpyDeviceToHost), "download uPrev");
    if (p.equation == Equation::GrayScott)
        ckCuda(cudaMemcpy(v, d_v, bytes, cudaMemcpyDeviceToHost), "download v");
}

void cudaShutdown() {
    if (g_a)    cudaFree(g_a);
    if (g_b)    cudaFree(g_b);
    if (g_c)    cudaFree(g_c);
    if (g_v)    cudaFree(g_v);
    if (g_v2)   cudaFree(g_v2);
    if (g_mask) cudaFree(g_mask);
    g_a = g_b = g_c = g_v = g_v2 = nullptr;
    g_mask = nullptr;
    g_cap = 0;
    g_maskVersion = -1;
}
