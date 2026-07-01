#pragma once
#include "Types.h"
#include <QImage>
#include <QString>
#include <vector>
#include <cstdint>

enum class Backend {
    CpuSingle,
    CpuMulti,
    Cuda
};

struct StepResult {
    double  milliseconds = 0.0;   // wall time for the whole step() call
    QString backendName;
};

// Owns the simulation state (the field and, for the wave equation, the previous
// field) and advances it with the chosen back-end. State is authoritative on the
// host; the CUDA back-end uploads/downloads around each step() call.
class Simulator {
public:
    void reset(const SimParams& p);                        // reallocate + seed
    void inject(const SimParams& p, float nx, float ny,
                float radiusCells, float amplitude);       // add a disturbance
    StepResult step(const SimParams& p, Backend backend);  // advance p.substeps

    QImage toImage(const SimParams& p) const;

    void saveState();      // snapshot for non-destructive benchmarking
    void restoreState();

    static bool    cudaSupported();
    static QString backendName(Backend b);

private:
    std::vector<float> m_u;       // current field
    std::vector<float> m_uPrev;   // previous field (wave equation)
    std::vector<float> m_next;    // scratch
    std::vector<float> m_savedU, m_savedPrev;
    int m_w = 0, m_h = 0;

    void stepCpu(const SimParams& p, bool multiThreaded);
#ifdef HAVE_CUDA
    void stepCuda(const SimParams& p);
#endif
};
