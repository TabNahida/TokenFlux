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
    std::size_t file_index = 0;
    std::vector<std::string> docs;
};

std::size_t derive_stream_target_bytes(std::uint64_t file_size, std::size_t worker_threads)
{
    const std::uint64_t min_target = 8ull * 1024ull * 1024ull;
    const std::uint64_t max_target = 128ull * 1024ull * 1024ull;
    if (worker_threads == 0)
    {
        worker_threads = 1;
    }
    if (file_size == 0)
    {
        return static_cast<std::size_t>(min_target);
    }
    std::uint64_t desired_parts = std::max<std::uint64_t>(1, std::min<std::uint64_t>(worker_threads * 2ull, 64ull));
    std::uint64_t target = (file_size + desired_parts - 1ull) / desired_parts;
    target = std::max<std::uint64_t>(target, min_target);
    target = std::min<std::uint64_t>(target, max_target);
    return static_cast<std::size_t>(target);
}

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

bool count_total_records(const Config &cfg, const std::vector<InputSource> &sources, uint64_t &total_records, std::string &err)
{
    total_records = 0;
    ProgressTracker scan_progress(static_cast<uint64_t>(sources.size()), "scanning", cfg.progress_interval_ms, "files");
    constexpr uint64_t scan_report_batch = 512;
    for (const auto &source : sources)
    {
        uint64_t local_docs = 0;
        uint64_t local_reported = 0;
        std::string local_err;
        bool ok = for_each_text_record(
            source.local_path, cfg.text_field,
            [&](const std::string &text) {
                if (!text.empty())
                {
                    ++local_docs;
                    uint64_t pending = local_docs - local_reported;
                    if (pending >= scan_report_batch)
                    {
                        scan_progress.add(0, pending);
                        local_reported = local_docs;
                    }
                }
            },
            local_err);
        if (!ok)
        {
            err = local_err.empty() ? ("failed to scan input file: " + source.source) : local_err;
            return false;
        }
        if (local_docs > local_reported)
        {
            scan_progress.add(0, local_docs - local_reported);
        }
        total_records += local_docs;
        scan_progress.add(1, 0);
    }
    scan_progress.finish();
    return true;
}
} // namespace

std::string chunk_path_for_id(const Config &cfg, std::size_t chunk_id)
{
    std::ostringstream oss;
    oss << cfg.chunk_dir << "/chunk_" << std::setw(8) << std::setfill('0') << chunk_id << ".cbk";
    return oss.str();
}

bool build_count_chunks(const Config &cfg, const std::vector<InputSource> &sources, std::size_t local_entry_cap,
                        const ProcessTextFn &process_text, ChunkBuildStats &stats, std::string &err)
{
    stats = {};
    if (sources.empty())
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

    uint64_t total_records_est = 0;
    if (cfg.prescan_records)
    {
        if (!count_total_records(cfg, sources, total_records_est, err))
        {
            return false;
        }
        if (total_records_est == 0)
        {
            err = "no text records found from inputs";
            return false;
        }
    }

    ProgressTracker progress(static_cast<uint64_t>(sources.size()), "processing", cfg.progress_interval_ms, "files");
    if (cfg.prescan_records && total_records_est > 0)
    {
        progress.set_total_docs(total_records_est);
    }

    std::deque<ChunkTask> queue;
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    bool read_done = false;

    std::atomic<std::size_t> next_chunk_id{0};
    std::atomic<uint64_t> docs_done{0};
    std::atomic<bool> had_error{false};
    std::vector<std::atomic<std::size_t>> pending_chunks(sources.size());
    std::vector<std::atomic<bool>> file_read_done(sources.size());
    std::vector<std::atomic<bool>> file_progress_done(sources.size());
    for (std::size_t i = 0; i < sources.size(); ++i)
    {
        pending_chunks[i].store(0, std::memory_order_relaxed);
        file_read_done[i].store(false, std::memory_order_relaxed);
        file_progress_done[i].store(false, std::memory_order_relaxed);
    }
    std::mutex err_mu;
    std::string shared_err;

    auto maybe_finish_file = [&](std::size_t file_index) {
        if (!file_read_done[file_index].load(std::memory_order_acquire))
        {
            return;
        }
        if (pending_chunks[file_index].load(std::memory_order_acquire) != 0)
        {
            return;
        }
        bool expected = false;
        if (file_progress_done[file_index].compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        {
            progress.add(1, 0);
        }
    };

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
        constexpr uint64_t process_report_batch = 128;
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
                progress.add(0, header.doc_count);
                pending_chunks[task.file_index].fetch_sub(1, std::memory_order_acq_rel);
                maybe_finish_file(task.file_index);
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
            uint64_t docs_reported = 0;
            std::size_t reduce_counter = 0;
            for (const auto &text : task.docs)
            {
                process_text(text, local_counts, docs, reduce_counter, local_entry_cap);
                uint64_t pending = docs - docs_reported;
                if (pending >= process_report_batch)
                {
                    progress.add(0, pending);
                    docs_reported = docs;
                }
            }
            if (docs > docs_reported)
            {
                progress.add(0, docs - docs_reported);
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
            pending_chunks[task.file_index].fetch_sub(1, std::memory_order_acq_rel);
            maybe_finish_file(task.file_index);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(cfg.threads);
    for (std::size_t i = 0; i < cfg.threads; ++i)
    {
        workers.emplace_back(worker);
    }

    auto enqueue_docs = [&](std::size_t file_index, std::vector<std::string> docs) -> bool {
        if (docs.empty())
        {
            return true;
        }
        ChunkTask task;
        task.chunk_id = next_chunk_id.fetch_add(1, std::memory_order_relaxed);
        task.file_index = file_index;
        task.docs = std::move(docs);

        std::unique_lock<std::mutex> lock(queue_mu);
        queue_cv.wait(lock, [&]() { return had_error.load() || queue.size() < queue_cap; });
        if (had_error.load())
        {
            return false;
        }
        pending_chunks[file_index].fetch_add(1, std::memory_order_acq_rel);
        queue.push_back(std::move(task));
        lock.unlock();
        queue_cv.notify_all();
        return true;
    };

    for (std::size_t file_index = 0; file_index < sources.size(); ++file_index)
    {
        if (had_error.load())
        {
            break;
        }
        const auto &source = sources[file_index];
        std::vector<std::string> batch;
        batch.reserve(cfg.records_per_chunk);
        std::size_t batch_bytes = 0;
        std::size_t target_bytes = derive_stream_target_bytes(source.file_size, cfg.threads);
        std::string local_err;
        bool ok = for_each_text_record(
            source.local_path, cfg.text_field,
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
                batch_bytes += text.size();
                if (batch.size() >= cfg.records_per_chunk || batch_bytes >= target_bytes)
                {
                    std::vector<std::string> flushed;
                    flushed.swap(batch);
                    batch_bytes = 0;
                    if (!enqueue_docs(file_index, std::move(flushed)))
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
            shared_err = local_err.empty() ? ("failed to read input file: " + source.source) : local_err;
            break;
        }
        if (!had_error.load() && !batch.empty())
        {
            if (!enqueue_docs(file_index, std::move(batch)))
            {
                had_error.store(true);
            }
        }
        file_read_done[file_index].store(true, std::memory_order_release);
        maybe_finish_file(file_index);
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
