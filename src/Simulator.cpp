#include "Simulator.h"
#include "Stencil.h"

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <algorithm>
#include <cmath>

#ifdef HAVE_CUDA
#include "CudaWave.h"
#endif

namespace {

// ---------------------------------------------------------------------------
// A small persistent thread pool for the multi-threaded CPU back-end.
//
// The previous version spawned std::thread objects for every substep, so a
// 100-step benchmark paid for ~100 * N thread creations. Here N-1 workers are
// created once and parked on a condition variable; each job splits the row
// range into small chunks that threads pull from an atomic counter (which also
// balances load when some rows are cheaper than others). The calling thread
// participates in the work instead of just waiting.
// ---------------------------------------------------------------------------
class RowPool {
public:
    RowPool() {
        unsigned hw = std::thread::hardware_concurrency();
        m_workers = (hw > 1) ? (int)hw - 1 : 0;   // the caller is the +1 thread
        m_threads.reserve(m_workers);
        for (int i = 0; i < m_workers; ++i)
            m_threads.emplace_back([this] { workerLoop(); });
    }

    ~RowPool() {
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_stop = true;
        }
        m_workCv.notify_all();
        for (auto& t : m_threads) t.join();
    }

    int threadCount() const { return m_workers + 1; }

    // Runs fn(yStart, yEnd) over [0, rows), split into chunks, in parallel.
    // Blocks until every chunk is done.
    void run(int rows, const std::function<void(int, int)>& fn) {
        if (rows <= 0) return;
        if (m_workers == 0) { fn(0, rows); return; }

        int chunks = std::min(rows, threadCount() * 4);
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_fn     = &fn;
            m_rows   = rows;
            m_chunks = chunks;
            m_nextChunk.store(0, std::memory_order_relaxed);
            m_active.store(m_workers, std::memory_order_relaxed);
            ++m_generation;
        }
        m_workCv.notify_all();

        work(fn, rows, chunks);                    // caller helps

        std::unique_lock<std::mutex> lk(m_mutex);
        m_doneCv.wait(lk, [this] { return m_active.load(std::memory_order_acquire) == 0; });
    }

private:
    void work(const std::function<void(int, int)>& fn, int rows, int chunks) {
        int per = (rows + chunks - 1) / chunks;
        int i;
        while ((i = m_nextChunk.fetch_add(1, std::memory_order_relaxed)) < chunks) {
            int ys = i * per;
            int ye = std::min(rows, ys + per);
            if (ys < ye) fn(ys, ye);
        }
    }

    void workerLoop() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int, int)>* fn;
            int rows, chunks;
            {
                std::unique_lock<std::mutex> lk(m_mutex);
                m_workCv.wait(lk, [&] { return m_stop || m_generation != seen; });
                if (m_stop) return;
                seen   = m_generation;
                fn     = m_fn;
                rows   = m_rows;
                chunks = m_chunks;
            }
            work(*fn, rows, chunks);
            if (m_active.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(m_mutex);   // pairs with the waiter
                m_doneCv.notify_all();
            }
        }
    }

    std::vector<std::thread> m_threads;
    std::mutex               m_mutex;
    std::condition_variable  m_workCv, m_doneCv;
    const std::function<void(int, int)>* m_fn = nullptr;
    int      m_rows = 0, m_chunks = 0, m_workers = 0;
    uint64_t m_generation = 0;
    std::atomic<int> m_nextChunk{0}, m_active{0};
    bool     m_stop = false;
};

RowPool& pool() {
    static RowPool p;   // created on first use, joined at program exit
    return p;
}

} // namespace

// ---------------------------------------------------------------------------

void Simulator::reset(const SimParams& p) {
    m_w = p.gridW;
    m_h = p.gridH;
    size_t n = (size_t)m_w * m_h;

    bool gs = (p.equation == Equation::GrayScott);
    m_u.assign(n, gs ? 1.0f : 0.0f);      // GS rest state: u = 1 everywhere
    m_uPrev.assign(n, gs ? 1.0f : 0.0f);
    m_next.assign(n, 0.0f);
    m_v.assign(n, 0.0f);
    m_vNext.assign(n, 0.0f);
    m_mask.assign(n, 0);

    buildScene(p);
    seed(p);
    ++m_maskVersion;
}

void Simulator::buildScene(const SimParams& p) {
    auto wallAt = [&](int x, int y) {
        if (x < 0 || x >= m_w || y < 0 || y >= m_h) return;
        int idx = y * m_w + x;
        m_mask[idx] = 1;
        bool gs = (p.equation == Equation::GrayScott);
        m_u[idx]     = gs ? 1.0f : 0.0f;
        m_uPrev[idx] = m_u[idx];
        m_v[idx]     = 0.0f;
    };

    switch (p.scene) {
    case 1: {   // Double slit: vertical wall with two gaps
        int cx        = m_w / 2;
        int slitHalf  = std::max(2, (int)(m_h * 0.025f));
        int y1        = (int)(m_h * 0.40f);
        int y2        = (int)(m_h * 0.60f);
        for (int y = 0; y < m_h; ++y) {
            if (std::abs(y - y1) <= slitHalf || std::abs(y - y2) <= slitHalf) continue;
            for (int dx = -1; dx <= 1; ++dx) wallAt(cx + dx, y);
        }
        break;
    }
    case 2: {   // Ring cavity: circular mirror with an opening on the right
        float cx = m_w * 0.5f, cy = m_h * 0.5f;
        float R  = 0.36f * (float)std::min(m_w, m_h);
        for (int y = 0; y < m_h; ++y) {
            for (int x = 0; x < m_w; ++x) {
                float dx = x - cx, dy = y - cy;
                float d  = std::sqrt(dx * dx + dy * dy);
                if (std::fabs(d - R) > 1.6f) continue;
                float ang = std::atan2(dy, dx);              // 0 points right
                if (std::fabs(ang) < 0.25f) continue;        // the opening
                wallAt(x, y);
            }
        }
        break;
    }
    case 3: {   // Pillar lattice: an 8x8 grid of small round obstacles
        int   nPerAxis = 8;
        float spacing  = 1.0f / (float)nPerAxis;
        float r        = std::max(2.0f, m_w * 0.014f);
        for (int j = 0; j < nPerAxis; ++j) {
            for (int i = 0; i < nPerAxis; ++i) {
                float cx = (i + 0.5f) * spacing * m_w;
                float cy = (j + 0.5f) * spacing * m_h;
                int ir = (int)r + 1;
                for (int dy = -ir; dy <= ir; ++dy)
                    for (int dx = -ir; dx <= ir; ++dx)
                        if ((float)(dx * dx + dy * dy) <= r * r)
                            wallAt((int)cx + dx, (int)cy + dy);
            }
        }
        break;
    }
    default:
        break;      // empty
    }
}

void Simulator::seed(const SimParams& p) {
    if (p.equation == Equation::GrayScott) {
        // A few v-patches so patterns start growing immediately.
        const float spots[5][2] = { {0.50f, 0.50f}, {0.34f, 0.40f}, {0.66f, 0.62f},
                                    {0.42f, 0.70f}, {0.62f, 0.32f} };
        for (auto& s : spots)
            inject(p, s[0], s[1], m_w * 0.018f, 1.0f);
        return;
    }
    // Wave / heat: a pulse. For the double slit, put it left of the wall so the
    // wavefront hits both slits.
    if (p.scene == 1) inject(p, 0.22f, 0.5f, m_w * 0.03f, 1.0f);
    else              inject(p, 0.5f,  0.5f, m_w * 0.03f, 1.0f);
}

void Simulator::inject(const SimParams& p, float nx, float ny,
                       float radiusCells, float amplitude) {
    if (m_u.empty()) return;
    int cx = (int)(nx * m_w);
    int cy = (int)(ny * m_h);
    int r  = std::max(1, (int)radiusCells);
    float r2 = (float)r * r;
    bool gs = (p.equation == Equation::GrayScott);

    for (int dy = -r; dy <= r; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= m_h) continue;
        for (int dx = -r; dx <= r; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= m_w) continue;
            float d2 = (float)(dx * dx + dy * dy);
            if (d2 > r2) continue;
            int idx = y * m_w + x;
            if (m_mask[idx]) continue;                       // never write into walls
            float g = amplitude * std::exp(-d2 / (0.5f * r2));
            if (gs) {
                // Seed the consumer chemical; eat a bit of the food locally.
                m_v[idx] = std::min(1.0f, m_v[idx] + g);
                m_u[idx] = std::max(0.0f, m_u[idx] - 0.5f * g);
            } else {
                m_u[idx] += g;
                // For the wave equation, start the bump at rest (uPrev == u) so
                // it radiates outward as a ripple instead of jerking.
                if (p.equation == Equation::Wave)
                    m_uPrev[idx] = m_u[idx];
            }
        }
    }
}

void Simulator::setWall(const SimParams& p, float nx, float ny,
                        float radiusCells, bool erase) {
    if (m_mask.empty()) return;
    int cx = (int)(nx * m_w);
    int cy = (int)(ny * m_h);
    int r  = std::max(1, (int)radiusCells);
    float r2 = (float)r * r;
    bool gs = (p.equation == Equation::GrayScott);

    for (int dy = -r; dy <= r; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= m_h) continue;
        for (int dx = -r; dx <= r; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= m_w) continue;
            if ((float)(dx * dx + dy * dy) > r2) continue;
            int idx = y * m_w + x;
            m_mask[idx] = erase ? 0 : 1;
            // Reset the field under the brush to the medium's rest state.
            m_u[idx]     = gs ? 1.0f : 0.0f;
            m_uPrev[idx] = m_u[idx];
            m_v[idx]     = 0.0f;
        }
    }
    ++m_maskVersion;
}

void Simulator::stepCpu(const SimParams& p, bool multiThreaded) {
    const uint8_t* mask = m_mask.data();

    for (int s = 0; s < p.substeps; ++s) {
        // Pointers are re-read each substep because the buffers swap below.
        auto rows = [&](int ys, int ye) {
            switch (p.equation) {
            case Equation::Wave: {
                float k = p.waveSpeed * p.waveSpeed;
                const float* u  = m_u.data();
                const float* up = m_uPrev.data();
                float* next     = m_next.data();
                for (int y = ys; y < ye; ++y)
                    for (int x = 0; x < p.gridW; ++x)
                        next[y * p.gridW + x] = sc_waveStep(u, up, mask, x, y,
                                                            p.gridW, p.gridH,
                                                            k, p.damping, p.boundary);
                break;
            }
            case Equation::Heat: {
                const float* u = m_u.data();
                float* next    = m_next.data();
                for (int y = ys; y < ye; ++y)
                    for (int x = 0; x < p.gridW; ++x)
                        next[y * p.gridW + x] = sc_heatStep(u, mask, x, y,
                                                            p.gridW, p.gridH,
                                                            p.diffusion, p.boundary);
                break;
            }
            case Equation::GrayScott: {
                const float* u = m_u.data();
                const float* v = m_v.data();
                float* nu      = m_next.data();
                float* nv      = m_vNext.data();
                for (int y = ys; y < ye; ++y)
                    for (int x = 0; x < p.gridW; ++x)
                        sc_gsStep(u, v, mask, x, y, p.gridW, p.gridH,
                                  p.du, p.dv, p.feed, p.kill, p.boundary,
                                  &nu[y * p.gridW + x], &nv[y * p.gridW + x]);
                break;
            }
            }
        };

        if (multiThreaded) pool().run(p.gridH, rows);
        else               rows(0, p.gridH);

        switch (p.equation) {
        case Equation::Wave:
            m_uPrev.swap(m_u);     // uPrev <- old u
            m_u.swap(m_next);      // u     <- next
            break;
        case Equation::Heat:
            m_u.swap(m_next);
            break;
        case Equation::GrayScott:
            m_u.swap(m_next);
            m_v.swap(m_vNext);
            break;
        }
    }
}

#ifdef HAVE_CUDA
void Simulator::stepCuda(const SimParams& p, bool tiled) {
    cudaStepField(p, m_u.data(), m_uPrev.data(), m_v.data(),
                  m_mask.data(), m_maskVersion, p.substeps, tiled);
}
#endif

StepResult Simulator::step(const SimParams& p, Backend backend) {
    StepResult res;
    auto t0 = std::chrono::high_resolution_clock::now();

    switch (backend) {
        case Backend::CpuSingle: stepCpu(p, false); break;
        case Backend::CpuMulti:  stepCpu(p, true);  break;
        case Backend::Cuda:
        case Backend::CudaTiled:
#ifdef HAVE_CUDA
            stepCuda(p, backend == Backend::CudaTiled);
#else
            stepCpu(p, true); backend = Backend::CpuMulti;
#endif
            break;
    }
    res.backendName = backendName(backend);

    auto t1 = std::chrono::high_resolution_clock::now();
    res.milliseconds = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

QImage Simulator::toImage(const SimParams& p) const {
    QImage img(m_w, m_h, QImage::Format_RGB32);
    int  diverging = (p.equation == Equation::Wave) ? 1 : 0;
    bool gs        = (p.equation == Equation::GrayScott);
    const uint32_t wallColor = 0xFF4A4B55u;

    for (int y = 0; y < m_h; ++y) {
        auto* line = reinterpret_cast<uint32_t*>(img.scanLine(y));
        for (int x = 0; x < m_w; ++x) {
            int idx = y * m_w + x;
            if (m_mask[idx]) { line[x] = wallColor; continue; }
            float v = gs ? m_v[idx] * 3.0f : m_u[idx];   // v peaks near ~0.35
            line[x] = sc_colorize(v, diverging, p.colorScheme);
        }
    }
    return img;
}

void Simulator::saveState() {
    m_savedU    = m_u;
    m_savedPrev = m_uPrev;
    m_savedV    = m_v;
}

void Simulator::restoreState() {
    if (!m_savedU.empty())    m_u     = m_savedU;
    if (!m_savedPrev.empty()) m_uPrev = m_savedPrev;
    if (!m_savedV.empty())    m_v     = m_savedV;
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
        case Backend::CudaTiled: return "CUDA (GPU, tiled)";
    }
    return "unknown";
}
