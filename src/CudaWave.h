#pragma once
#include "Types.h"
#include <cstdint>

// Host-side interface to the CUDA back-end (implemented in CudaWave.cu).
// Qt-free so it can be included from both host and device translation units.

bool cudaAvailable();

// Advances the field(s) by `substeps` simulation steps on the GPU.
// u/uPrev/v are host buffers of gridW*gridH floats: they are uploaded, iterated
// on the device (rotating device pointers between steps, no per-step copies),
// and the updated fields are written back. Which of them are used depends on
// p.equation (wave: u+uPrev, heat: u, Gray-Scott: u+v).
//
// mask marks obstacle cells; it is uploaded only when maskVersion changes.
// tiled selects the shared-memory-tiled kernels instead of the naive ones.
void cudaStepField(const SimParams& p, float* u, float* uPrev, float* v,
                   const uint8_t* mask, int maskVersion, int substeps, bool tiled);

void cudaShutdown();
