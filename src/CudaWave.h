#pragma once
#include "Types.h"

// Host-side interface to the CUDA back-end (implemented in CudaWave.cu).
// Qt-free so it can be included from both host and device translation units.

bool cudaAvailable();

// Advances the field by `substeps` simulation steps on the GPU.
// u and uPrev are host buffers of gridW*gridH floats: they are uploaded,
// iterated on the device, and the updated fields are written back.
// The elapsed GPU wall time (upload + steps + download) is written to *elapsedMs.
void cudaStepField(const SimParams& p, float* u, float* uPrev,
                   int substeps, double* elapsedMs);

void cudaShutdown();
