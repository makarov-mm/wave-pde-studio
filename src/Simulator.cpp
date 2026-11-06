#include "Simulator.h"

#include "Colormap.h"
#include "RowPool.h"
#include "Stencil.h"

#ifdef HAVE_CUDA
#include "CudaWave.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>

namespace {

constexpr std::uint32_t kWallColor = 0xFF4A4B55u;

// Rest state of the main field: 1 for Gray-Scott's "food" u, 0 for waves and heat.
constexpr float restU(Equation equation) {
    return equation == Equation::GrayScott ? stencil::kWallU : 0.0f;
}

// Calls cell(i, x, y, interior) for every cell of rows [rowBegin, rowEnd).
// Cells on the grid edge get interior == false and go through the boundary
// rule; the long run of interior cells in between has no boundary branches,
// which keeps the hot loop simple enough for the compiler to vectorize.
template <class Cell>
void sweepRows(int w, int h, int rowBegin, int rowEnd, Cell&& cell) {
    for (int y = rowBegin; y < rowEnd; ++y) {
        const int row = y * w;
        if (y == 0 || y == h - 1 || w < 3) {
            for (int x = 0; x < w; ++x)
                cell(row + x, x, y, false);
            continue;
        }
        cell(row, 0, y, false);
        for (int x = 1; x < w - 1; ++x)
            cell(row + x, x, y, true);
        cell(row + w - 1, w - 1, y, false);
    }
}

void runRows(bool parallel, int rows, const std::function<void(int, int)>& body) {
    if (parallel)
        RowPool::instance().run(rows, body);
    else
        body(0, rows);
}

} // namespace

Simulator::Simulator() = default;
Simulator::~Simulator() = default;

void Simulator::reset(const SimParams& p) {
    m_w = p.gridW;
    m_h = p.gridH;
    const std::size_t cells = static_cast<std::size_t>(m_w) * static_cast<std::size_t>(m_h);

    const float rest = restU(p.equation);
    m_u.assign(cells, rest);
    m_uPrev.assign(cells, rest);
    m_next.assign(cells, 0.0f);
    m_v.assign(cells, 0.0f);
    m_vNext.assign(cells, 0.0f);
    m_mask.assign(cells, 0);

    buildScene(p);
    seed(p);
    ++m_maskVersion;
}

void Simulator::buildScene(const SimParams& p) {
    const float rest = restU(p.equation);
    const auto wallAt = [&](int x, int y) {
        if (x < 0 || x >= m_w || y < 0 || y >= m_h)
            return;
        const int i = y * m_w + x;
        m_mask[i]  = 1;
        m_u[i]     = rest;
        m_uPrev[i] = rest;
        m_v[i]     = 0.0f;
    };

    switch (p.scene) {
    case Scene::DoubleSlit: {   // vertical wall with two gaps
        const int cx       = m_w / 2;
        const int slitHalf = std::max(2, static_cast<int>(static_cast<float>(m_h) * 0.025f));
        const int y1       = static_cast<int>(static_cast<float>(m_h) * 0.40f);
        const int y2       = static_cast<int>(static_cast<float>(m_h) * 0.60f);
        for (int y = 0; y < m_h; ++y) {
            if (std::abs(y - y1) <= slitHalf || std::abs(y - y2) <= slitHalf)
                continue;
            for (int dx = -1; dx <= 1; ++dx)
                wallAt(cx + dx, y);
        }
        break;
    }
    case Scene::RingCavity: {   // circular mirror with an opening on the right
        const float cx = static_cast<float>(m_w) * 0.5f;
        const float cy = static_cast<float>(m_h) * 0.5f;
        const float radius = 0.36f * static_cast<float>(std::min(m_w, m_h));
        for (int y = 0; y < m_h; ++y) {
            for (int x = 0; x < m_w; ++x) {
                const float dx = static_cast<float>(x) - cx;
                const float dy = static_cast<float>(y) - cy;
                if (std::fabs(std::sqrt(dx * dx + dy * dy) - radius) > 1.6f)
                    continue;
                if (std::fabs(std::atan2(dy, dx)) < 0.25f)   // the opening; angle 0 points right
                    continue;
                wallAt(x, y);
            }
        }
        break;
    }
    case Scene::PillarLattice: {   // an 8 x 8 grid of small round obstacles
        constexpr int perAxis = 8;
        const float spacing = 1.0f / static_cast<float>(perAxis);
        const float radius  = std::max(2.0f, static_cast<float>(m_w) * 0.014f);
        const int   reach   = static_cast<int>(radius) + 1;
        for (int j = 0; j < perAxis; ++j) {
            for (int i = 0; i < perAxis; ++i) {
                const int cx = static_cast<int>((static_cast<float>(i) + 0.5f) * spacing * static_cast<float>(m_w));
                const int cy = static_cast<int>((static_cast<float>(j) + 0.5f) * spacing * static_cast<float>(m_h));
                for (int dy = -reach; dy <= reach; ++dy)
                    for (int dx = -reach; dx <= reach; ++dx)
                        if (static_cast<float>(dx * dx + dy * dy) <= radius * radius)
                            wallAt(cx + dx, cy + dy);
            }
        }
        break;
    }
    case Scene::Empty:
        break;
    }
}

void Simulator::seed(const SimParams& p) {
    const float w = static_cast<float>(m_w);

    if (p.equation == Equation::GrayScott) {
        // A few v patches so that patterns start growing right away.
        constexpr float spots[][2] = { {0.50f, 0.50f}, {0.34f, 0.40f}, {0.66f, 0.62f},
                                       {0.42f, 0.70f}, {0.62f, 0.32f} };
        for (const auto& [x, y] : spots)
            inject(p, x, y, w * 0.018f, 1.0f);
        return;
    }

    // Wave / heat: one pulse. Behind the double slit it starts left of the wall,
    // so that the wavefront reaches both slits.
    const float x = (p.scene == Scene::DoubleSlit) ? 0.22f : 0.5f;
    inject(p, x, 0.5f, w * 0.03f, 1.0f);
}

template <class Fn>
void Simulator::forEachCellInDisk(float nx, float ny, float radiusCells, Fn&& fn) {
    const int   cx = static_cast<int>(nx * static_cast<float>(m_w));
    const int   cy = static_cast<int>(ny * static_cast<float>(m_h));
    const int   r  = std::max(1, static_cast<int>(radiusCells));
    const float r2 = static_cast<float>(r) * static_cast<float>(r);

    for (int y = std::max(0, cy - r); y <= std::min(m_h - 1, cy + r); ++y) {
        for (int x = std::max(0, cx - r); x <= std::min(m_w - 1, cx + r); ++x) {
            const int dx = x - cx;
            const int dy = y - cy;
            const float d2 = static_cast<float>(dx * dx + dy * dy);
            if (d2 <= r2)
                fn(y * m_w + x, d2, r2);
        }
    }
}

void Simulator::inject(const SimParams& p, float nx, float ny, float radiusCells, float amplitude) {
    const bool grayScott = p.equation == Equation::GrayScott;
    const bool wave      = p.equation == Equation::Wave;

    forEachCellInDisk(nx, ny, radiusCells, [&](int i, float d2, float r2) {
        if (m_mask[i])
            return;   // never write into walls
        const float g = amplitude * std::exp(-d2 / (0.5f * r2));
        if (grayScott) {
            // Seed the consumer chemical and eat a bit of the food locally.
            m_v[i] = std::min(1.0f, m_v[i] + g);
            m_u[i] = std::max(0.0f, m_u[i] - 0.5f * g);
        } else {
            m_u[i] += g;
            // Waves: start the bump at rest (uPrev == u) so that it radiates
            // outward as a ripple instead of jerking.
            if (wave)
                m_uPrev[i] = m_u[i];
        }
    });
}

void Simulator::paintWall(const SimParams& p, float nx, float ny, float radiusCells, bool erase) {
    const float rest = restU(p.equation);

    forEachCellInDisk(nx, ny, radiusCells, [&](int i, float, float) {
        m_mask[i] = erase ? 0 : 1;
        // The field under the brush returns to the medium's rest state.
        m_u[i]     = rest;
        m_uPrev[i] = rest;
        m_v[i]     = 0.0f;
    });
    ++m_maskVersion;
}

void Simulator::stepCpu(const SimParams& p, bool parallel) {
    const int      w        = m_w;
    const int      h        = m_h;
    const Boundary boundary = p.boundary;
    const std::uint8_t* mask = m_mask.data();

    // Laplacian of `field` at one cell: the boundary-free formula for interior cells.
    // The cell updates below are computed for wall cells too and then discarded:
    // a branch-free select lets the compiler vectorize the interior loop.
    const auto lap = [=](const float* field, int i, int x, int y, bool interior) {
        return interior ? stencil::laplacianInterior(field, i, w)
                        : stencil::laplacian(field, x, y, w, h, boundary);
    };

    for (int s = 0; s < p.substeps; ++s) {
        switch (p.equation) {
        case Equation::Wave: {
            const float  k       = p.waveSpeed * p.waveSpeed;
            const float  damping = p.damping;
            const float* u       = m_u.data();
            const float* uPrev   = m_uPrev.data();
            float*       next    = m_next.data();
            runRows(parallel, h, [=](int rowBegin, int rowEnd) {
                sweepRows(w, h, rowBegin, rowEnd, [=](int i, int x, int y, bool interior) {
                    const float updated = stencil::waveUpdate(u[i], uPrev[i], lap(u, i, x, y, interior), k, damping);
                    next[i] = mask[i] ? stencil::kWallWave : updated;
                });
            });
            m_uPrev.swap(m_u);   // uPrev <- old u
            m_u.swap(m_next);    // u     <- next
            break;
        }
        case Equation::Heat: {
            const float  alpha = p.diffusion;
            const float* u     = m_u.data();
            float*       next  = m_next.data();
            runRows(parallel, h, [=](int rowBegin, int rowEnd) {
                sweepRows(w, h, rowBegin, rowEnd, [=](int i, int x, int y, bool interior) {
                    const float updated = stencil::heatUpdate(u[i], lap(u, i, x, y, interior), alpha);
                    next[i] = mask[i] ? stencil::kWallHeat : updated;
                });
            });
            m_u.swap(m_next);
            break;
        }
        case Equation::GrayScott: {
            const float  du = p.du, dv = p.dv, feed = p.feed, kill = p.kill;
            const float* u     = m_u.data();
            const float* v     = m_v.data();
            float*       nextU = m_next.data();
            float*       nextV = m_vNext.data();
            runRows(parallel, h, [=](int rowBegin, int rowEnd) {
                sweepRows(w, h, rowBegin, rowEnd, [=](int i, int x, int y, bool interior) {
                    const stencil::GrayScottCell updated = stencil::grayScottUpdate(
                        u[i], v[i], lap(u, i, x, y, interior), lap(v, i, x, y, interior), du, dv, feed, kill);
                    const bool wall = mask[i] != 0;
                    nextU[i] = wall ? stencil::kWallU : updated.u;
                    nextV[i] = wall ? stencil::kWallV : updated.v;
                });
            });
            m_u.swap(m_next);
            m_v.swap(m_vNext);
            break;
        }
        }
    }
}

#ifdef HAVE_CUDA
void Simulator::stepCuda(const SimParams& p, bool tiled) {
    if (!m_cuda)
        m_cuda = std::make_unique<CudaSolver>();
    m_cuda->step(p, m_w, m_h, m_u.data(), m_uPrev.data(), m_v.data(), m_mask.data(), m_maskVersion, tiled);
}
#endif

StepResult Simulator::step(const SimParams& p, Backend backend) {
    const auto start = std::chrono::steady_clock::now();

    switch (backend) {
    case Backend::CpuSingle:
        stepCpu(p, false);
        break;
    case Backend::CpuMulti:
        stepCpu(p, true);
        break;
    case Backend::Cuda:
    case Backend::CudaTiled:
#ifdef HAVE_CUDA
        stepCuda(p, backend == Backend::CudaTiled);
#else
        stepCpu(p, true);
        backend = Backend::CpuMulti;
#endif
        break;
    }

    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
    return { elapsed.count(), backend };
}

QImage Simulator::toImage(const SimParams& p) const {
    QImage image(m_w, m_h, QImage::Format_RGB32);
    if (image.isNull())
        return image;

    uchar* const    bits      = image.bits();   // detach once, before the threads start
    const qsizetype stride    = image.bytesPerLine();
    const bool      diverging = p.equation == Equation::Wave;
    const bool      grayScott = p.equation == Equation::GrayScott;

    RowPool::instance().run(m_h, [&](int rowBegin, int rowEnd) {
        for (int y = rowBegin; y < rowEnd; ++y) {
            auto* line = reinterpret_cast<std::uint32_t*>(bits + y * stride);
            for (int x = 0; x < m_w; ++x) {
                const int i = y * m_w + x;
                if (m_mask[i]) {
                    line[x] = kWallColor;
                    continue;
                }
                const float value = grayScott ? m_v[i] * 3.0f : m_u[i];   // v peaks near ~0.35
                line[x] = colormap::colorize(value, diverging, p.palette);
            }
        }
    });
    return image;
}

Simulator::Snapshot Simulator::snapshot() const {
    return { m_u, m_uPrev, m_v };
}

void Simulator::restore(const Snapshot& snapshot) {
    if (snapshot.u.size() != m_u.size())
        return;   // taken before a grid change
    m_u     = snapshot.u;
    m_uPrev = snapshot.uPrev;
    m_v     = snapshot.v;
}

QString Simulator::cudaUnavailableReason() {
#ifdef HAVE_CUDA
    return QString::fromStdString(CudaSolver::unavailableReason());
#else
    return QStringLiteral("this build has no CUDA support (USE_CUDA=OFF)");
#endif
}

QString Simulator::backendName(Backend backend) {
    switch (backend) {
    case Backend::CpuSingle: return QStringLiteral("CPU (single-threaded)");
    case Backend::CpuMulti:  return QStringLiteral("CPU (multi-threaded)");
    case Backend::Cuda:      return QStringLiteral("CUDA (GPU)");
    case Backend::CudaTiled: return QStringLiteral("CUDA (GPU, tiled)");
    }
    return QStringLiteral("unknown");
}
