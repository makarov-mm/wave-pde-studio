#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

// A persistent thread pool for fork-join loops over rows.
//
// hardware_concurrency() - 1 workers are started once and park on a condition
// variable between jobs; the calling thread does its share of the work too.
// Rows are handed out in small chunks from an atomic counter, which also
// balances the load when some rows are cheaper than others.
class RowPool {
public:
    static RowPool& instance();

    RowPool();
    RowPool(const RowPool&) = delete;
    RowPool& operator=(const RowPool&) = delete;

    int threadCount() const { return static_cast<int>(m_workers.size()) + 1; }

    // Calls body(rowBegin, rowEnd) over disjoint chunks covering [0, rows) and
    // returns once every chunk is done. Not reentrant: body must not call run().
    void run(int rows, const std::function<void(int, int)>& body);

private:
    void workerLoop(std::stop_token stop);
    void processChunks();

    std::mutex                  m_mutex;
    std::condition_variable_any m_wake;   // a new job was published (or stop was requested)
    std::condition_variable     m_done;   // the last worker finished its share

    // The current job; written under m_mutex before m_generation is bumped.
    const std::function<void(int, int)>* m_body = nullptr;
    int           m_rows       = 0;
    int           m_chunks     = 0;
    std::uint64_t m_generation = 0;

    std::atomic<int> m_nextChunk{0};
    std::atomic<int> m_busyWorkers{0};

    // Declared last so the threads are stopped and joined before anything they use is destroyed.
    std::vector<std::jthread> m_workers;
};
