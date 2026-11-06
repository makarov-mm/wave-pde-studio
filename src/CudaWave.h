#pragma once
#include "Types.h"

#include <cstdint>
#include <memory>
#include <string>

// Host-side interface to the CUDA back-ends (implemented in CudaWave.cu).
// Free of CUDA and Qt headers, so plain C++ translation units can include it.
class CudaSolver {
public:
    // Empty if a CUDA device can run this build's kernels; otherwise the reason why not
    // (no driver, no device, or a GPU this build was not compiled for).
    static std::string unavailableReason();

    CudaSolver();
    ~CudaSolver();
    CudaSolver(const CudaSolver&) = delete;
    CudaSolver& operator=(const CudaSolver&) = delete;

    // Advances the fields by p.substeps steps on the GPU. u/uPrev/v are host buffers
    // of w*h floats: the ones p.equation uses (wave: u + uPrev, heat: u, Gray-Scott:
    // u + v) are uploaded, iterated on the device with rotating buffers (no copies
    // between steps) and downloaded again. The mask is uploaded only when maskVersion
    // changes. `tiled` selects the shared-memory kernels.
    // Throws std::runtime_error on any CUDA error.
    void step(const SimParams& p, int w, int h, float* u, float* uPrev, float* v,
              const std::uint8_t* mask, int maskVersion, bool tiled);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
