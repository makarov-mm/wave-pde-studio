#pragma once
#include "Types.h"

#include <QImage>
#include <QString>

#include <cstdint>
#include <memory>
#include <vector>

class CudaSolver;

enum class Backend : int {
    CpuSingle,
    CpuMulti,
    Cuda,        // one thread per cell, plain global-memory reads
    CudaTiled    // same, but the stencil reads from a shared-memory tile + halo
};

struct StepResult {
    double  milliseconds = 0.0;   // wall time of the whole step() call, transfers included
    Backend backend      = Backend::CpuSingle;
};

// Owns the simulation state (field u, previous field for the wave equation,
// second field v for Gray-Scott, and the obstacle mask) and advances it with the
// chosen back-end. The host copy is authoritative: the CUDA back-ends upload the
// state and download the result around each step() call.
class Simulator {
public:
    // The fields a benchmark needs to rewind; walls are not touched by stepping.
    struct Snapshot {
        std::vector<float> u, uPrev, v;
    };

    Simulator();
    ~Simulator();
    Simulator(const Simulator&) = delete;
    Simulator& operator=(const Simulator&) = delete;

    void reset(const SimParams& p);    // reallocate, build the scene, seed the field

    // Brush operations at normalized [0, 1] coordinates.
    void inject(const SimParams& p, float nx, float ny, float radiusCells, float amplitude);
    void paintWall(const SimParams& p, float nx, float ny, float radiusCells, bool erase);

    // Advances p.substeps steps. Throws std::runtime_error if a GPU back-end fails.
    StepResult step(const SimParams& p, Backend backend);

    QImage toImage(const SimParams& p) const;

    Snapshot snapshot() const;
    void     restore(const Snapshot& snapshot);

    // Why the CUDA back-ends can't be used here, or an empty string if they can.
    static QString cudaUnavailableReason();
    static QString backendName(Backend backend);

private:
    template <class Fn>
    void forEachCellInDisk(float nx, float ny, float radiusCells, Fn&& fn);

    void buildScene(const SimParams& p);
    void seed(const SimParams& p);
    void stepCpu(const SimParams& p, bool parallel);

    std::vector<float>        m_u;       // current field (wave/heat) or Gray-Scott chemical u
    std::vector<float>        m_uPrev;   // previous field (wave equation only)
    std::vector<float>        m_next;    // scratch
    std::vector<float>        m_v;       // Gray-Scott chemical v
    std::vector<float>        m_vNext;   // scratch for v
    std::vector<std::uint8_t> m_mask;    // 1 = wall/obstacle
    int m_w = 0;
    int m_h = 0;
    int m_maskVersion = 0;               // bumped whenever the mask changes (GPU cache key)

#ifdef HAVE_CUDA
    void stepCuda(const SimParams& p, bool tiled);

    std::unique_ptr<CudaSolver> m_cuda;  // created on first GPU step
#endif
};
