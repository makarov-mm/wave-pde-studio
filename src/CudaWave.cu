// CudaWave.cu
// GPU back-ends: one thread per grid cell advances the finite-difference stencil.
// The cell math comes from Stencil.h, shared with the CPU path, and the build
// disables FMA contraction, so the GPU fields match the CPU fields bit for bit.
//
// Two kernel families are provided per equation:
//   * naive - every thread reads its five stencil values straight from global
//             memory (the L1/L2 caches absorb most of the redundancy);
//   * tiled - each block first stages a (kTile + 2)^2 tile of the field, including
//             a one-cell halo, in shared memory and computes the stencil from it.
//             This is the classic stencil optimization; the app benchmarks both.

#include "CudaWave.h"
#include "Stencil.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr int kTile = 16;                    // block side; the tiled kernels rely on blockDim == kTile
using Tile = float[kTile + 2][kTile + 2];    // a block's cells plus a one-cell halo

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error (") + what + "): " + cudaGetErrorString(error));
}

// Owning pointer to a device array.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(m_data); }

    // Replaces the buffer with an uninitialized one of `count` elements.
    void allocate(std::size_t count) {
        cudaFree(m_data);
        m_data = nullptr;
        check(cudaMalloc(&m_data, count * sizeof(T)), "cudaMalloc");
    }

    T* get() const { return m_data; }

private:
    T* m_data = nullptr;
};

// ---------------------------------------------------------------------------
// Naive kernels: the shared cell math, reading neighbors from global memory.
// ---------------------------------------------------------------------------

__global__ void waveKernel(float* __restrict__ next, const float* __restrict__ u,
                           const float* __restrict__ uPrev, const std::uint8_t* __restrict__ mask,
                           int w, int h, float k, float damping, Boundary boundary) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h)
        return;
    const int i = y * w + x;
    next[i] = mask[i] ? stencil::kWallWave
                      : stencil::waveUpdate(u[i], uPrev[i], stencil::laplacian(u, x, y, w, h, boundary),
                                            k, damping);
}

__global__ void heatKernel(float* __restrict__ next, const float* __restrict__ u,
                           const std::uint8_t* __restrict__ mask,
                           int w, int h, float alpha, Boundary boundary) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h)
        return;
    const int i = y * w + x;
    next[i] = mask[i] ? stencil::kWallHeat
                      : stencil::heatUpdate(u[i], stencil::laplacian(u, x, y, w, h, boundary), alpha);
}

__global__ void grayScottKernel(float* __restrict__ nextU, float* __restrict__ nextV,
                                const float* __restrict__ u, const float* __restrict__ v,
                                const std::uint8_t* __restrict__ mask, int w, int h,
                                float du, float dv, float feed, float kill, Boundary boundary) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h)
        return;
    const int i = y * w + x;
    const stencil::GrayScottCell cell =
        mask[i] ? stencil::GrayScottCell{stencil::kWallU, stencil::kWallV}
                : stencil::grayScottUpdate(u[i], v[i],
                                           stencil::laplacian(u, x, y, w, h, boundary),
                                           stencil::laplacian(v, x, y, w, h, boundary),
                                           du, dv, feed, kill);
    nextU[i] = cell.u;
    nextV[i] = cell.v;
}

// ---------------------------------------------------------------------------
// Tiled kernels: stage the tile + halo in shared memory, then apply the same
// cell math to values read from the tile.
// ---------------------------------------------------------------------------

// Every thread loads its own cell; threads on the block edge also load the halo.
// Cells outside the grid receive boundary values, so partial tiles work as well.
__device__ void loadTile(Tile& tile, const float* field, int x, int y, int w, int h, Boundary boundary) {
    const int tx = static_cast<int>(threadIdx.x) + 1;
    const int ty = static_cast<int>(threadIdx.y) + 1;
    tile[ty][tx] = stencil::sample(field, x, y, w, h, boundary);
    if (tx == 1)     tile[ty][0]         = stencil::sample(field, x - 1, y, w, h, boundary);
    if (tx == kTile) tile[ty][kTile + 1] = stencil::sample(field, x + 1, y, w, h, boundary);
    if (ty == 1)     tile[0][tx]         = stencil::sample(field, x, y - 1, w, h, boundary);
    if (ty == kTile) tile[kTile + 1][tx] = stencil::sample(field, x, y + 1, w, h, boundary);
}

__device__ float tileCenter(const Tile& tile) {
    return tile[threadIdx.y + 1][threadIdx.x + 1];
}

__device__ float tileLaplacian(const Tile& tile) {
    const int tx = static_cast<int>(threadIdx.x) + 1;
    const int ty = static_cast<int>(threadIdx.y) + 1;
    return stencil::laplacian5(tile[ty][tx], tile[ty][tx - 1], tile[ty][tx + 1], tile[ty - 1][tx], tile[ty + 1][tx]);
}

__global__ void waveKernelTiled(float* __restrict__ next, const float* __restrict__ u,
                                const float* __restrict__ uPrev, const std::uint8_t* __restrict__ mask,
                                int w, int h, float k, float damping, Boundary boundary) {
    __shared__ Tile tile;
    const int x = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * kTile + static_cast<int>(threadIdx.y);
    loadTile(tile, u, x, y, w, h, boundary);
    __syncthreads();
    if (x >= w || y >= h)
        return;
    const int i = y * w + x;
    next[i] = mask[i] ? stencil::kWallWave
                      : stencil::waveUpdate(tileCenter(tile), uPrev[i], tileLaplacian(tile), k, damping);
}

__global__ void heatKernelTiled(float* __restrict__ next, const float* __restrict__ u,
                                const std::uint8_t* __restrict__ mask,
                                int w, int h, float alpha, Boundary boundary) {
    __shared__ Tile tile;
    const int x = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * kTile + static_cast<int>(threadIdx.y);
    loadTile(tile, u, x, y, w, h, boundary);
    __syncthreads();
    if (x >= w || y >= h)
        return;
    const int i = y * w + x;
    next[i] = mask[i] ? stencil::kWallHeat : stencil::heatUpdate(tileCenter(tile), tileLaplacian(tile), alpha);
}

__global__ void grayScottKernelTiled(float* __restrict__ nextU, float* __restrict__ nextV,
                                     const float* __restrict__ u, const float* __restrict__ v,
                                     const std::uint8_t* __restrict__ mask, int w, int h,
                                     float du, float dv, float feed, float kill, Boundary boundary) {
    __shared__ Tile tileU;
    __shared__ Tile tileV;
    const int x = static_cast<int>(blockIdx.x) * kTile + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * kTile + static_cast<int>(threadIdx.y);
    loadTile(tileU, u, x, y, w, h, boundary);
    loadTile(tileV, v, x, y, w, h, boundary);
    __syncthreads();
    if (x >= w || y >= h)
        return;
    const int i = y * w + x;
    const stencil::GrayScottCell cell =
        mask[i] ? stencil::GrayScottCell{stencil::kWallU, stencil::kWallV}
                : stencil::grayScottUpdate(tileCenter(tileU), tileCenter(tileV),
                                           tileLaplacian(tileU), tileLaplacian(tileV),
                                           du, dv, feed, kill);
    nextU[i] = cell.u;
    nextV[i] = cell.v;
}

} // namespace

// ---------------------------------------------------------------------------

struct CudaSolver::Impl {
    DeviceBuffer<float>        u, uPrev, next, v, vNext;
    DeviceBuffer<std::uint8_t> mask;
    std::size_t capacity    = 0;    // cells every buffer can hold
    int         maskVersion = -1;   // version of the mask on the device (-1: none)

    void reserve(std::size_t cells) {
        if (cells <= capacity)
            return;
        capacity = 0;   // stays 0 if an allocation throws
        for (DeviceBuffer<float>* buffer : {&u, &uPrev, &next, &v, &vNext})
            buffer->allocate(cells);
        mask.allocate(cells);
        capacity    = cells;
        maskVersion = -1;   // the new mask buffer is uninitialized
    }
};

CudaSolver::CudaSolver() : m_impl(std::make_unique<Impl>()) {}

CudaSolver::~CudaSolver() = default;

std::string CudaSolver::unavailableReason() {
    int devices = 0;
    if (const cudaError_t error = cudaGetDeviceCount(&devices); error != cudaSuccess)
        return std::string("no usable CUDA driver or device (") + cudaGetErrorString(error) + ")";
    if (devices == 0)
        return "no CUDA device found";

    // A build for another compute capability has no kernel image for this GPU,
    // and every launch would fail.
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, waveKernel) != cudaSuccess) {
        cudaGetLastError();   // clear the error state
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
            return "the CUDA kernels of this build cannot run on this GPU";
        return std::string("this build has no kernels for ") + properties.name + " (compute capability " +
               std::to_string(properties.major) + "." + std::to_string(properties.minor) +
               "); rebuild with -DCUDA_ARCH=" + std::to_string(properties.major * 10 + properties.minor);
    }
    return {};
}

void CudaSolver::step(const SimParams& p, int w, int h, float* u, float* uPrev, float* v,
                      const std::uint8_t* mask, int maskVersion, bool tiled) {
    Impl& d = *m_impl;
    const std::size_t cells = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::size_t bytes = cells * sizeof(float);
    d.reserve(cells);

    // The mask changes rarely (scene switches, wall painting), so it is uploaded
    // only when its version differs from the one already on the device.
    if (maskVersion != d.maskVersion) {
        check(cudaMemcpy(d.mask.get(), mask, cells, cudaMemcpyHostToDevice), "upload mask");
        d.maskVersion = maskVersion;
    }

    const bool wave      = p.equation == Equation::Wave;
    const bool grayScott = p.equation == Equation::GrayScott;

    // Device pointers rotate between steps; the buffers themselves never move.
    float* dU     = d.u.get();
    float* dPrev  = d.uPrev.get();
    float* dNext  = d.next.get();
    float* dV     = d.v.get();
    float* dVNext = d.vNext.get();
    const std::uint8_t* dMask = d.mask.get();

    // Upload, iterate on the device, download. The caller times the whole call,
    // so the reported cost honestly includes the transfers.
    check(cudaMemcpy(dU, u, bytes, cudaMemcpyHostToDevice), "upload u");
    if (wave)
        check(cudaMemcpy(dPrev, uPrev, bytes, cudaMemcpyHostToDevice), "upload uPrev");
    if (grayScott)
        check(cudaMemcpy(dV, v, bytes, cudaMemcpyHostToDevice), "upload v");

    const dim3  block(kTile, kTile);
    const dim3  grid(static_cast<unsigned>((w + kTile - 1) / kTile), static_cast<unsigned>((h + kTile - 1) / kTile));
    const float k = p.waveSpeed * p.waveSpeed;

    for (int s = 0; s < p.substeps; ++s) {
        switch (p.equation) {
        case Equation::Wave: {
            if (tiled)
                waveKernelTiled<<<grid, block>>>(dNext, dU, dPrev, dMask, w, h, k, p.damping, p.boundary);
            else
                waveKernel<<<grid, block>>>(dNext, dU, dPrev, dMask, w, h, k, p.damping, p.boundary);
            float* const oldPrev = dPrev;   // prev <- u, u <- next, next <- old prev
            dPrev = dU;
            dU    = dNext;
            dNext = oldPrev;
            break;
        }
        case Equation::Heat:
            if (tiled)
                heatKernelTiled<<<grid, block>>>(dNext, dU, dMask, w, h, p.diffusion, p.boundary);
            else
                heatKernel<<<grid, block>>>(dNext, dU, dMask, w, h, p.diffusion, p.boundary);
            std::swap(dU, dNext);
            break;
        case Equation::GrayScott:
            if (tiled)
                grayScottKernelTiled<<<grid, block>>>(dNext, dVNext, dU, dV, dMask, w, h,
                                                      p.du, p.dv, p.feed, p.kill, p.boundary);
            else
                grayScottKernel<<<grid, block>>>(dNext, dVNext, dU, dV, dMask, w, h,
                                                 p.du, p.dv, p.feed, p.kill, p.boundary);
            std::swap(dU, dNext);
            std::swap(dV, dVNext);
            break;
        }
        check(cudaGetLastError(), "kernel launch");
    }

    // cudaMemcpy waits for the kernels, so execution errors surface here too.
    check(cudaMemcpy(u, dU, bytes, cudaMemcpyDeviceToHost), "download u");
    if (wave)
        check(cudaMemcpy(uPrev, dPrev, bytes, cudaMemcpyDeviceToHost), "download uPrev");
    if (grayScott)
        check(cudaMemcpy(v, dV, bytes, cudaMemcpyDeviceToHost), "download v");
}
