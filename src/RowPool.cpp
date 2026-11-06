#include "RowPool.h"

#include <algorithm>

RowPool& RowPool::instance() {
    static RowPool pool;   // started on first use, stopped and joined at program exit
    return pool;
}

RowPool::RowPool() {
    const unsigned hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    m_workers.reserve(hardwareThreads - 1);   // the caller is the remaining thread
    for (unsigned i = 1; i < hardwareThreads; ++i)
        m_workers.emplace_back([this](std::stop_token stop) { workerLoop(stop); });
}

void RowPool::run(int rows, const std::function<void(int, int)>& body) {
    if (rows <= 0)
        return;
    if (m_workers.empty()) {
        body(0, rows);
        return;
    }

    {
        std::lock_guard lock(m_mutex);
        m_body   = &body;
        m_rows   = rows;
        m_chunks = std::min(rows, threadCount() * 4);
        m_nextChunk.store(0, std::memory_order_relaxed);
        m_busyWorkers.store(static_cast<int>(m_workers.size()), std::memory_order_relaxed);
        ++m_generation;
    }
    m_wake.notify_all();

    processChunks();   // the caller helps

    std::unique_lock lock(m_mutex);
    m_done.wait(lock, [this] { return m_busyWorkers.load(std::memory_order_acquire) == 0; });
}

void RowPool::processChunks() {
    const int rowsPerChunk = (m_rows + m_chunks - 1) / m_chunks;
    for (int chunk; (chunk = m_nextChunk.fetch_add(1, std::memory_order_relaxed)) < m_chunks;) {
        const int rowBegin = chunk * rowsPerChunk;
        const int rowEnd   = std::min(m_rows, rowBegin + rowsPerChunk);
        if (rowBegin < rowEnd)
            (*m_body)(rowBegin, rowEnd);
    }
}

void RowPool::workerLoop(std::stop_token stop) {
    std::uint64_t seenGeneration = 0;
    for (;;) {
        {
            std::unique_lock lock(m_mutex);
            if (!m_wake.wait(lock, stop, [&] { return m_generation != seenGeneration; }))
                return;   // stop requested
            seenGeneration = m_generation;
        }

        processChunks();

        // The release half pairs with the caller's acquire load: everything this
        // worker wrote is visible once run() returns.
        if (m_busyWorkers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard lock(m_mutex);   // so the notification can't slip in before the caller waits
            m_done.notify_all();
        }
    }
}
