#pragma once
#include "Types.h"
#include <QImage>
#include <QString>
#include <vector>
#include <cstdint>

enum class Backend {
    CpuSingle,
    CpuMulti,
    Cuda,        // one thread per cell, plain global-memory reads
    CudaTiled    // same, but the stencil reads from a shared-memory tile + halo
};

struct StepResult {
    double  milliseconds = 0.0;   // wall time for the whole step() call
    QString backendName;
};

// Owns the simulation state (field u, previous field for the wave equation,
// second field v for Gray-Scott, and the obstacle mask) and advances it with
// the chosen back-end. State is authoritative on the host; the CUDA back-ends
// upload/download around each step() call.
class Simulator {
public:
    void reset(const SimParams& p);                        // reallocate + scene + seed
    void inject(const SimParams& p, float nx, float ny,
                float radiusCells, float amplitude);       // add a disturbance
    void setWall(const SimParams& p, float nx, float ny,
                 float radiusCells, bool erase);           // paint / erase obstacles
    StepResult step(const SimParams& p, Backend backend);  // advance p.substeps

    QImage toImage(const SimParams& p) const;

    void saveState();      // snapshot for non-destructive benchmarking
    void restoreState();

    static bool    cudaSupported();
    static QString backendName(Backend b);

private:
    std::vector<float>   m_u;       // current field (wave/heat) or GS chemical u
    std::vector<float>   m_uPrev;   // previous field (wave equation only)
    std::vector<float>   m_next;    // scratch
    std::vector<float>   m_v;       // Gray-Scott chemical v
    std::vector<float>   m_vNext;   // scratch for v
    std::vector<uint8_t> m_mask;    // 1 = wall/obstacle
    std::vector<float>   m_savedU, m_savedPrev, m_savedV;
    int m_w = 0, m_h = 0;
    int m_maskVersion = 0;          // bumped whenever the mask changes (GPU cache key)

    void buildScene(const SimParams& p);
    void seed(const SimParams& p);
    void stepCpu(const SimParams& p, bool multiThreaded);
#ifdef HAVE_CUDA
    void stepCuda(const SimParams& p, bool tiled);
#endif
};
