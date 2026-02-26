#include "train_io.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tokenflux_lib.hpp"

namespace
{
struct ChunkTask
{
    std::size_t chunk_id = 0;
    std::vector<std::string> docs;
};

std::unordered_map<std::string, uint64_t> reduce_top_k_u64(std::unordered_map<std::string, uint64_t> &counts,
                                                           std::size_t top_k)
{
    if (top_k == 0 || counts.size() <= top_k)
    {
        return std::move(counts);
    }
    std::vector<std::pair<std::string, uint64_t>> vec;
    vec.reserve(counts.size());
    for (auto &kv : counts)
    {
        vec.emplace_back(std::move(kv.first), kv.second);
    }
    counts.clear();
    auto nth = vec.begin() + static_cast<std::ptrdiff_t>(top_k);
    std::nth_element(vec.begin(), nth, vec.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
    vec.resize(top_k);
    std::unordered_map<std::string, uint64_t> reduced;
    reduced.reserve(vec.size() * 13 / 10 + 8);
    for (auto &kv : vec)
    {
        reduced.emplace(std::move(kv.first), kv.second);
    }
    return reduced;
}
} // namespace

std::string chunk_path_for_id(const Config &cfg, std::size_t chunk_id)
{
    std::ostringstream oss;
    oss << cfg.chunk_dir << "/chunk_" << std::setw(8) << std::setfill('0') << chunk_id << ".cbk";
    return oss.str();
}

bool build_count_chunks(const Config &cfg, const std::vector<std::string> &files, std::size_t local_entry_cap,
                        const ProcessTextFn &process_text, ChunkBuildStats &stats, std::string &err)
{
    stats = {};
    if (files.empty())
    {
        err = "no files to process";
        return false;
    }
    if (!process_text)
    {
        err = "internal: process_text callback is empty";
        return false;
    }

    std::size_t queue_cap = cfg.queue_capacity;
    if (queue_cap == 0)
    {
        queue_cap = std::max<std::size_t>(cfg.threads * 4, 16);
    }

    ProgressTracker progress(0, "processing", cfg.progress_interval_ms);

    std::deque<ChunkTask> queue;
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    bool read_done = false;

    std::atomic<std::size_t> next_chunk_id{0};
    std::atomic<uint64_t> docs_done{0};
    std::atomic<bool> had_error{false};
    std::mutex err_mu;
    std::string shared_err;

    auto take_task = [&](ChunkTask &task) -> bool {
        std::unique_lock<std::mutex> lock(queue_mu);
        queue_cv.wait(lock, [&]() { return had_error.load() || !queue.empty() || read_done; });
        if (had_error.load())
        {
            return false;
        }
        if (queue.empty())
        {
            return false;
        }
        task = std::move(queue.front());
        queue.pop_front();
        lock.unlock();
        queue_cv.notify_all();
        return true;
    };

    auto worker = [&]() {
        while (true)
        {
            ChunkTask task;
            if (!take_task(task))
            {
                break;
            }

            std::string out_path = chunk_path_for_id(cfg, task.chunk_id);
            if (cfg.resume && std::filesystem::exists(out_path))
            {
                ChunkHeader header;
                if (!read_chunk_header(out_path, header))
                {
                    std::lock_guard<std::mutex> lock(err_mu);
                    had_error.store(true);
                    if (shared_err.empty())
                    {
                        shared_err = "failed to read chunk header: " + out_path;
                    }
                    queue_cv.notify_all();
                    break;
                }
                docs_done.fetch_add(header.doc_count, std::memory_order_relaxed);
                progress.add(1, header.doc_count);
                continue;
            }

            LocalCountMap local_counts;
            std::size_t reserve_hint = cfg.top_k;
            if (local_entry_cap > 0 && (reserve_hint == 0 || reserve_hint > local_entry_cap))
            {
                reserve_hint = local_entry_cap;
            }
            if (reserve_hint == 0)
            {
                reserve_hint = 1024;
            }
            local_counts.reserve(reserve_hint * 2 + 16);

            uint64_t docs = 0;
            std::size_t reduce_counter = 0;
            for (const auto &text : task.docs)
            {
                process_text(text, local_counts, docs, reduce_counter, local_entry_cap);
            }

            std::size_t keep = cfg.top_k;
            if (local_entry_cap > 0 && (keep == 0 || keep > local_entry_cap))
            {
                keep = local_entry_cap;
            }
            if (keep > 0)
            {
                local_counts = reduce_top_k(local_counts, keep);
            }

            if (!write_chunk_file(out_path, local_counts, docs))
            {
                std::lock_guard<std::mutex> lock(err_mu);
                had_error.store(true);
                if (shared_err.empty())
                {
                    shared_err = "failed to write chunk: " + out_path;
                }
                queue_cv.notify_all();
                break;
            }

            docs_done.fetch_add(docs, std::memory_order_relaxed);
            progress.add(1, docs);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(cfg.threads);
    for (std::size_t i = 0; i < cfg.threads; ++i)
    {
        workers.emplace_back(worker);
    }

    auto enqueue_docs = [&](std::vector<std::string> docs) -> bool {
        if (docs.empty())
        {
            return true;
        }
        ChunkTask task;
        task.chunk_id = next_chunk_id.fetch_add(1, std::memory_order_relaxed);
        task.docs = std::move(docs);

        std::unique_lock<std::mutex> lock(queue_mu);
        queue_cv.wait(lock, [&]() { return had_error.load() || queue.size() < queue_cap; });
        if (had_error.load())
        {
            return false;
        }
        queue.push_back(std::move(task));
        lock.unlock();
        progress.add_total(1);
        queue_cv.notify_all();
        return true;
    };

    std::vector<std::string> batch;
    batch.reserve(cfg.records_per_chunk);
    for (const auto &path : files)
    {
        if (had_error.load())
        {
            break;
        }
        std::string local_err;
        bool ok = for_each_text_record(
            path, cfg.text_field,
            [&](const std::string &text) {
                if (had_error.load())
                {
                    return;
                }
                if (text.empty())
                {
                    return;
                }
                batch.push_back(text);
                if (batch.size() >= cfg.records_per_chunk)
                {
                    std::vector<std::string> flushed;
                    flushed.swap(batch);
                    if (!enqueue_docs(std::move(flushed)))
                    {
                        had_error.store(true);
                    }
                }
            },
            local_err);
        if (!ok)
        {
            std::lock_guard<std::mutex> lock(err_mu);
            had_error.store(true);
            shared_err = local_err.empty() ? ("failed to read input file: " + path) : local_err;
            break;
        }
    }

    if (!had_error.load() && !batch.empty())
    {
        if (!enqueue_docs(std::move(batch)))
        {
            had_error.store(true);
        }
    }

    {
        std::lock_guard<std::mutex> lock(queue_mu);
        read_done = true;
    }
    queue_cv.notify_all();

    for (auto &t : workers)
    {
        t.join();
    }
    progress.finish();

    if (had_error.load())
    {
        err = shared_err.empty() ? "processing failed" : shared_err;
        return false;
    }

    stats.total_chunks = next_chunk_id.load(std::memory_order_relaxed);
    stats.total_docs = docs_done.load(std::memory_order_relaxed);
    if (stats.total_chunks == 0)
    {
        err = "no text records found from inputs";
        return false;
    }
    return true;
}

bool merge_count_chunks(const Config &cfg, std::size_t total_chunks, std::size_t global_entry_cap,
                        GlobalCountMap &global_counts, uint64_t &total_docs, std::string &err)
{
    total_docs = 0;
    global_counts.clear();
    if (total_chunks == 0)
    {
        err = "no chunks to merge";
        return false;
    }
    global_counts.reserve(total_chunks * 1024);

    ProgressTracker merge_progress(total_chunks, "merging", cfg.progress_interval_ms);
    for (std::size_t chunk_id = 0; chunk_id < total_chunks; ++chunk_id)
    {
        std::string path = chunk_path_for_id(cfg, chunk_id);
        if (!merge_chunk_file(path, global_counts, &total_docs))
        {
            err = "failed to read chunk: " + path;
            return false;
        }
        if (global_entry_cap > 0 && global_counts.size() > global_entry_cap)
        {
            global_counts = reduce_top_k_u64(global_counts, global_entry_cap);
        }
        merge_progress.add(1, 0);
    }
    merge_progress.finish();
    return true;
}
