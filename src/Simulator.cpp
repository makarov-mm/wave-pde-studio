#include "Simulator.h"
#include "Stencil.h"

#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>
#include <cstring>
#include <cmath>

#ifdef HAVE_CUDA
#include "CudaWave.h"
#endif

namespace {

// Advance a horizontal band of rows [yStart, yEnd) by one wave step.
void waveRows(const SimParams& p, float* next, const float* u, const float* uPrev,
              int yStart, int yEnd) {
    float k = p.waveSpeed * p.waveSpeed;
    for (int y = yStart; y < yEnd; ++y)
        for (int x = 0; x < p.gridW; ++x)
            next[y * p.gridW + x] = sc_waveStep(u, uPrev, x, y, p.gridW, p.gridH, k, p.damping);
}

void heatRows(const SimParams& p, float* next, const float* u,
              int yStart, int yEnd) {
    for (int y = yStart; y < yEnd; ++y)
        for (int x = 0; x < p.gridW; ++x)
            next[y * p.gridW + x] = sc_heatStep(u, x, y, p.gridW, p.gridH, p.diffusion);
}

} // namespace

void Simulator::reset(const SimParams& p) {
    m_w = p.gridW;
    m_h = p.gridH;
    size_t n = (size_t)m_w * m_h;
    m_u.assign(n, 0.0f);
    m_uPrev.assign(n, 0.0f);
    m_next.assign(n, 0.0f);

    // Seed a central pulse so there is something moving right away.
    inject(p, 0.5f, 0.5f, m_w * 0.03f, (p.equation == Equation::Wave) ? 1.0f : 1.0f);
}

void Simulator::inject(const SimParams& p, float nx, float ny,
                       float radiusCells, float amplitude) {
    if (m_u.empty()) return;
    int cx = (int)(nx * m_w);
    int cy = (int)(ny * m_h);
    int r  = std::max(1, (int)radiusCells);
    float r2 = (float)r * r;

    for (int dy = -r; dy <= r; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= m_h) continue;
        for (int dx = -r; dx <= r; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= m_w) continue;
            float d2 = (float)(dx * dx + dy * dy);
            if (d2 > r2) continue;
            float g = amplitude * std::exp(-d2 / (0.5f * r2));
            int idx = y * m_w + x;
            m_u[idx] += g;
            // For the wave equation, start the bump at rest (uPrev == u) so it
            // radiates outward as a ripple instead of jerking.
            if (p.equation == Equation::Wave)
                m_uPrev[idx] = m_u[idx];
        }
    }
}

void Simulator::stepCpu(const SimParams& p, bool multiThreaded) {
    bool wave = (p.equation == Equation::Wave);

    for (int s = 0; s < p.substeps; ++s) {
        if (!multiThreaded) {
            if (wave) waveRows(p, m_next.data(), m_u.data(), m_uPrev.data(), 0, m_h);
            else      heatRows(p, m_next.data(), m_u.data(), 0, m_h);
        } else {
            unsigned hw = std::thread::hardware_concurrency();
            if (hw == 0) hw = 4;
            int nThreads = (int)hw;
            std::vector<std::thread> pool;
            int rowsPer = (m_h + nThreads - 1) / nThreads;
            for (int t = 0; t < nThreads; ++t) {
                int ys = t * rowsPer;
                int ye = std::min(m_h, ys + rowsPer);
                if (ys >= ye) break;
                if (wave)
                    pool.emplace_back(waveRows, std::cref(p), m_next.data(),
                                      m_u.data(), m_uPrev.data(), ys, ye);
                else
                    pool.emplace_back(heatRows, std::cref(p), m_next.data(),
                                      m_u.data(), ys, ye);
            }
            for (auto& th : pool) th.join();
        }

        if (wave) {
            m_uPrev.swap(m_u);     // uPrev <- old u
            m_u.swap(m_next);      // u     <- next
        } else {
            m_u.swap(m_next);
        }
    }
}

#ifdef HAVE_CUDA
void Simulator::stepCuda(const SimParams& p) {
    double ms = 0.0;
    cudaStepField(p, m_u.data(), m_uPrev.data(), p.substeps, &ms);
    // (timing captured by caller via wall clock; ms is available if needed)
    (void)ms;
}
#endif

StepResult Simulator::step(const SimParams& p, Backend backend) {
    StepResult res;
    auto t0 = std::chrono::high_resolution_clock::now();

    switch (backend) {
        case Backend::CpuSingle: stepCpu(p, false); res.backendName = "CPU (single-threaded)"; break;
        case Backend::CpuMulti:  stepCpu(p, true);  res.backendName = "CPU (multi-threaded)";  break;
        case Backend::Cuda:
#ifdef HAVE_CUDA
            stepCuda(p); res.backendName = "CUDA (GPU)";
#else
            stepCpu(p, true); res.backendName = "CPU (multi-threaded)";
#endif
            break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.milliseconds = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

QImage Simulator::toImage(const SimParams& p) const {
    QImage img(m_w, m_h, QImage::Format_RGB32);
    int eq = (int)p.equation;
    for (int y = 0; y < m_h; ++y) {
        auto* line = reinterpret_cast<uint32_t*>(img.scanLine(y));
        for (int x = 0; x < m_w; ++x)
            line[x] = sc_colorize(m_u[y * m_w + x], eq, p.colorScheme);
    }
    return img;
}

void Simulator::saveState() {
    m_savedU    = m_u;
    m_savedPrev = m_uPrev;
}

void Simulator::restoreState() {
    if (!m_savedU.empty())    m_u     = m_savedU;
    if (!m_savedPrev.empty()) m_uPrev = m_savedPrev;
}

bool Simulator::cudaSupported() {
#ifdef HAVE_CUDA
    return cudaAvailable();
#else
    return false;
#endif
}

QString Simulator::backendName(Backend b) {
    switch (b) {
        case Backend::CpuSingle: return "CPU (single-threaded)";
        case Backend::CpuMulti:  return "CPU (multi-threaded)";
        case Backend::Cuda:      return "CUDA (GPU)";
    }
    return "unknown";
}
