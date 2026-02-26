#include "tokenize_pipeline.hpp"

#include "tokenize_common.hpp"
#include "tokenize_tokenizer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize
{

static bool get_file_stat(const std::string &path, std::uint64_t &file_size, std::int64_t &mtime)
{
    std::error_code ec;
    file_size = std::filesystem::file_size(path, ec);
    if (ec)
    {
        return false;
    }
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec)
    {
        return false;
    }
    mtime = std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
    return true;
}

static std::string make_tokenizer_fingerprint(const std::string &path)
{
    std::error_code ec;
    std::filesystem::path abs_path = std::filesystem::absolute(path, ec);
    std::uint64_t sz = 0;
    std::int64_t mt = 0;
    if (!get_file_stat(path, sz, mt))
    {
        return {};
    }
    std::ostringstream oss;
    oss << normalize_path_for_compare(abs_path.string()) << "|" << sz << "|" << mt;
    return oss.str();
}

static std::string make_run_signature(const PartSignature &sig)
{
    std::ostringstream oss;
    oss << "tokenizer_fingerprint=" << sig.tokenizer_fingerprint << ";";
    oss << "text_field=" << sig.text_field << ";";
    oss << "min_chars=" << sig.min_chars << ";";
    oss << "max_chars=" << sig.max_chars << ";";
    oss << "bos_id=" << sig.bos_id << ";";
    oss << "eos_id=" << sig.eos_id << ";";
    oss << "dtype_bytes=" << sig.dtype_bytes;
    return oss.str();
}

static bool parse_completion_record_line(const std::string &line, std::string &path_out, CompletionRecord &record_out)
{
    std::istringstream iss(line);
    std::string path;
    std::uint64_t source_size = 0;
    std::int64_t source_mtime = 0;
    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::uint64_t num_tokens = 0;
    if (!(iss >> std::quoted(path) >> source_size >> source_mtime >> num_docs >> num_skipped >> num_tokens))
    {
        return false;
    }
    path_out = std::move(path);
    record_out.source_size = source_size;
    record_out.source_mtime = source_mtime;
    record_out.result.num_docs = num_docs;
    record_out.result.num_skipped = num_skipped;
    record_out.result.num_tokens = num_tokens;
    record_out.result.reused = true;
    return true;
}

static bool init_completion_records(const std::filesystem::path &path, const std::string &run_signature, bool resume,
                                    std::unordered_map<std::string, CompletionRecord> &records, std::string &err)
{
    records.clear();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
    {
        err = "failed to create completion dir: " + path.parent_path().string();
        return false;
    }

    bool rewrite_file = !resume;
    if (resume && std::filesystem::exists(path))
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            err = "failed to open completion list: " + path.string();
            return false;
        }
        std::string first_line;
        if (!std::getline(in, first_line))
        {
            rewrite_file = true;
        }
        else if (first_line != "signature=" + run_signature)
        {
            rewrite_file = true;
        }
        else
        {
            std::string line;
            while (std::getline(in, line))
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                if (line.empty() || line[0] == '#')
                {
                    continue;
                }
                std::string source_path;
                CompletionRecord rec;
                if (!parse_completion_record_line(line, source_path, rec))
                {
                    continue;
                }
                records[normalize_path_for_compare(source_path)] = rec;
            }
        }
    }
    else if (!resume)
    {
        rewrite_file = true;
    }

    if (rewrite_file || !std::filesystem::exists(path))
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            err = "failed to initialize completion list: " + path.string();
            return false;
        }
        out << "signature=" << run_signature << "\n";
        out << "# path size mtime docs skipped tokens\n";
        out.flush();
        if (!out)
        {
            err = "failed to flush completion list: " + path.string();
            return false;
        }
        records.clear();
    }

    return true;
}

static bool append_completion_record(const std::filesystem::path &path, const FileTask &task, const PartResult &result,
                                     std::mutex &mu, std::string &err)
{
    std::lock_guard<std::mutex> lock(mu);
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out)
    {
        err = "failed to append completion list: " + path.string();
        return false;
    }
    out << std::quoted(task.path) << "\t" << task.file_size << "\t" << task.file_mtime << "\t" << result.num_docs
        << "\t" << result.num_skipped << "\t" << result.num_tokens << "\n";
    out.flush();
    if (!out)
    {
        err = "failed to flush completion list: " + path.string();
        return false;
    }
    return true;
}

static bool completion_is_reusable(const FileTask &task, const std::unordered_map<std::string, CompletionRecord> &records,
                                   PartResult &result)
{
    auto it = records.find(task.normalized_path);
    if (it == records.end())
    {
        return false;
    }
    if (it->second.source_size != task.file_size || it->second.source_mtime != task.file_mtime)
    {
        return false;
    }
    result = it->second.result;
    result.reused = true;
    return true;
}

static std::string now_string()
{
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

static std::size_t utf8_char_count(const std::string &s)
{
    std::size_t i = 0;
    std::size_t n = 0;
    while (i < s.size())
    {
        std::size_t prev = i;
        std::uint32_t cp = 0;
        if (!next_codepoint(s, i, cp))
        {
            break;
        }
        if (i <= prev)
        {
            break;
        }
        ++n;
    }
    return n;
}

static bool prescan_total_docs(const std::vector<FileTask> &tasks, const Args &args, std::uint64_t &total_docs,
                               std::string &err)
{
    total_docs = 0;
    ProgressTracker scan_progress(tasks.size(), "scanning", 1000);
    constexpr std::uint64_t report_docs_batch = 512;

    for (const auto &task : tasks)
    {
        std::uint64_t local_docs = 0;
        std::uint64_t local_reported = 0;
        std::string local_err;
        bool ok = for_each_text_record(
            task.path, args.text_field,
            [&](const std::string &incoming_text) {
                if (incoming_text.empty())
                {
                    return;
                }
                std::size_t chars = utf8_char_count(incoming_text);
                if (chars < args.min_chars)
                {
                    return;
                }
                ++local_docs;
                std::uint64_t pending = local_docs - local_reported;
                if (pending >= report_docs_batch)
                {
                    scan_progress.add(0, pending);
                    local_reported = local_docs;
                }
            },
            local_err);
        if (!ok)
        {
            err = local_err.empty() ? ("failed to scan input file: " + task.path) : local_err;
            return false;
        }
        if (local_docs > local_reported)
        {
            scan_progress.add(0, local_docs - local_reported);
        }
        total_docs += local_docs;
        scan_progress.add(1, 0);
    }

    scan_progress.finish();
    return true;
}

static std::string shard_name(std::size_t idx)
{
    std::ostringstream oss;
    oss << "train_" << std::setw(6) << std::setfill('0') << idx << ".bin";
    return oss.str();
}

static bool parse_shard_index(const std::string &name, std::size_t &idx)
{
    if (!starts_with(name, "train_") || !ends_with(name, ".bin"))
    {
        return false;
    }
    std::string mid = name.substr(6, name.size() - 10);
    if (mid.empty())
    {
        return false;
    }
    std::uint64_t x = 0;
    if (!parse_u64_arg(mid, x))
    {
        return false;
    }
    if (x > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        return false;
    }
    idx = static_cast<std::size_t>(x);
    return true;
}

static bool remove_old_shards(const std::filesystem::path &out_dir);

class ShardWriter
{
  public:
    ShardWriter(std::filesystem::path out_dir, std::uint32_t dtype_bytes, std::uint64_t max_tokens_per_shard)
        : out_dir_(std::move(out_dir)), dtype_bytes_(dtype_bytes), max_tokens_per_shard_(max_tokens_per_shard)
    {
    }

    bool initialize(bool resume_existing, std::string &err)
    {
        std::lock_guard<std::mutex> lock(mu_);
        return initialize_locked(resume_existing, err);
    }

    bool append_tokens(const std::vector<std::uint32_t> &tokens, std::string &err)
    {
        if (tokens.empty())
        {
            return true;
        }
        std::lock_guard<std::mutex> lock(mu_);
        std::size_t offset = 0;
        while (offset < tokens.size())
        {
            if (!out_.is_open())
            {
                if (!open_new_shard_locked(err))
                {
                    return false;
                }
            }
            if (current_tokens_ >= max_tokens_per_shard_)
            {
                if (!rotate_shard_locked(err))
                {
                    return false;
                }
            }
            std::uint64_t remaining = max_tokens_per_shard_ - current_tokens_;
            std::size_t take = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(tokens.size() - offset)));
            if (!write_slice_locked(tokens.data() + offset, take, err))
            {
                return false;
            }
            current_tokens_ += static_cast<std::uint64_t>(take);
            total_tokens_ += static_cast<std::uint64_t>(take);
            offset += take;
        }
        return true;
    }

    bool finalize(std::vector<ShardInfo> &shards, std::uint64_t &total_tokens, std::string &err)
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!close_current_locked(true, err))
        {
            return false;
        }
        if (finalized_shards_.empty())
        {
            err = "no shards written";
            return false;
        }
        shards = finalized_shards_;
        total_tokens = total_tokens_;
        return true;
    }

    std::uint64_t total_tokens() const
    {
        return total_tokens_;
    }

  private:
    bool initialize_locked(bool resume_existing, std::string &err)
    {
        finalized_shards_.clear();
        total_tokens_ = 0;
        current_tokens_ = 0;
        shard_idx_ = 0;
        current_path_.clear();
        if (out_.is_open())
        {
            out_.close();
        }

        if (!resume_existing)
        {
            remove_old_shards(out_dir_);
            return open_new_shard_locked(err);
        }

        std::vector<std::pair<std::size_t, std::filesystem::path>> existing;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(out_dir_, ec); !ec && it != std::filesystem::directory_iterator();
             it.increment(ec))
        {
            if (!it->is_regular_file())
            {
                continue;
            }
            std::size_t idx = 0;
            std::string name = it->path().filename().string();
            if (parse_shard_index(name, idx))
            {
                existing.emplace_back(idx, it->path());
            }
        }
        if (ec)
        {
            err = "failed to list shard dir: " + out_dir_.string();
            return false;
        }
        if (existing.empty())
        {
            return open_new_shard_locked(err);
        }

        std::sort(existing.begin(), existing.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        for (std::size_t i = 0; i < existing.size(); ++i)
        {
            std::error_code sec;
            std::uint64_t bytes = std::filesystem::file_size(existing[i].second, sec);
            if (sec)
            {
                err = "failed to stat shard: " + existing[i].second.string();
                return false;
            }
            if (bytes % static_cast<std::uint64_t>(dtype_bytes_) != 0)
            {
                err = "corrupt shard file (size not aligned to dtype): " + existing[i].second.string();
                return false;
            }
            std::uint64_t tokens = bytes / static_cast<std::uint64_t>(dtype_bytes_);
            total_tokens_ += tokens;
            if (i + 1 < existing.size())
            {
                finalized_shards_.push_back({existing[i].second.filename().string(), tokens});
            }
            else
            {
                shard_idx_ = existing[i].first;
                current_path_ = existing[i].second;
                current_tokens_ = tokens;
            }
        }

        out_.open(current_path_, std::ios::binary | std::ios::app);
        if (!out_)
        {
            err = "failed to open shard for append: " + current_path_.string();
            return false;
        }
        return true;
    }

    bool open_new_shard_locked(std::string &err)
    {
        current_path_ = out_dir_ / shard_name(shard_idx_);
        out_.open(current_path_, std::ios::binary | std::ios::trunc);
        current_tokens_ = 0;
        if (!out_)
        {
            err = "failed to open shard for write: " + current_path_.string();
            return false;
        }
        return true;
    }

    bool close_current_locked(bool record, std::string &err)
    {
        if (!out_.is_open())
        {
            return true;
        }
        out_.flush();
        if (!out_)
        {
            err = "failed to flush shard: " + current_path_.string();
            return false;
        }
        out_.close();
        if (record)
        {
            finalized_shards_.push_back({current_path_.filename().string(), current_tokens_});
        }
        current_tokens_ = 0;
        return true;
    }

    bool rotate_shard_locked(std::string &err)
    {
        if (!close_current_locked(true, err))
        {
            return false;
        }
        ++shard_idx_;
        return open_new_shard_locked(err);
    }

    bool write_slice_locked(const std::uint32_t *tokens, std::size_t count, std::string &err)
    {
        if (count == 0)
        {
            return true;
        }
        if (dtype_bytes_ == 2)
        {
            tmp16_.clear();
            tmp16_.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                std::uint32_t v = tokens[i];
                if (v > std::numeric_limits<std::uint16_t>::max())
                {
                    err = "token id overflow for uint16 shards";
                    return false;
                }
                tmp16_.push_back(static_cast<std::uint16_t>(v));
            }
            out_.write(reinterpret_cast<const char *>(tmp16_.data()),
                       static_cast<std::streamsize>(tmp16_.size() * sizeof(std::uint16_t)));
        }
        else if (dtype_bytes_ == 4)
        {
            out_.write(reinterpret_cast<const char *>(tokens), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
        }
        else
        {
            err = "unsupported dtype bytes";
            return false;
        }
        if (!out_)
        {
            err = "failed to write shard: " + current_path_.string();
            return false;
        }
        return true;
    }

    std::filesystem::path out_dir_;
    std::uint32_t dtype_bytes_ = 2;
    std::uint64_t max_tokens_per_shard_ = 0;
    std::size_t shard_idx_ = 0;
    std::uint64_t current_tokens_ = 0;
    std::uint64_t total_tokens_ = 0;
    std::filesystem::path current_path_;
    std::ofstream out_;
    std::vector<std::uint16_t> tmp16_;
    std::vector<ShardInfo> finalized_shards_;
    std::mutex mu_;
};

static bool process_file_to_shards(const FileTask &task, const Args &args, const TokenizerEncoder &tokenizer,
                                   const PartSignature &sig, ShardWriter &shard_writer, std::size_t encode_threads,
                                   PartResult &result, const std::function<void(std::uint64_t)> &report_docs,
                                   std::string &err)
{
    std::size_t flush_threshold_tokens = 256 * 1024;
    std::size_t effective_cache_max_entries = args.cache_max_entries;
    if (args.max_memory_mb > 0)
    {
        std::uint64_t budget_bytes = static_cast<std::uint64_t>(args.max_memory_mb) * 1024ull * 1024ull;
        std::uint64_t buffer_budget = std::max<std::uint64_t>(budget_bytes / 4ull, 1ull << 20);
        std::size_t per_token_bytes = sizeof(std::uint32_t);
        std::size_t derived_flush = static_cast<std::size_t>(buffer_budget / std::max<std::size_t>(per_token_bytes, 1));
        flush_threshold_tokens = std::max<std::size_t>(64 * 1024, derived_flush);
        if (effective_cache_max_entries > 0)
        {
            std::size_t derived_cache_cap = static_cast<std::size_t>((budget_bytes * 3ull / 4ull) / 128ull);
            if (derived_cache_cap == 0)
            {
                derived_cache_cap = 1;
            }
            effective_cache_max_entries = std::min<std::size_t>(effective_cache_max_entries, derived_cache_cap);
        }
    }

    if (encode_threads <= 1)
    {
        std::vector<std::uint32_t> write_buffer;
        write_buffer.reserve(std::min<std::size_t>(flush_threshold_tokens, 1 << 20));
        std::unordered_map<std::string, std::vector<std::uint32_t>> cache;
        if (effective_cache_max_entries > 0)
        {
            const std::size_t reserve_n = std::min<std::size_t>(effective_cache_max_entries, 1 << 16);
            cache.reserve(reserve_n);
        }

        auto flush_buffer = [&]() -> bool {
            if (write_buffer.empty())
            {
                return true;
            }
            if (!shard_writer.append_tokens(write_buffer, err))
            {
                return false;
            }
            write_buffer.clear();
            return true;
        };

        constexpr std::uint64_t report_docs_batch = 128;
        std::uint64_t docs_pending_report = 0;
        bool callback_ok = true;
        bool read_ok = for_each_text_record(
            task.path, args.text_field,
            [&](const std::string &incoming_text) {
                if (!callback_ok)
                {
                    return;
                }
                if (incoming_text.empty())
                {
                    return;
                }
                std::string text = incoming_text;
                std::size_t chars = utf8_char_count(text);
                if (chars < args.min_chars)
                {
                    ++result.num_skipped;
                    return;
                }
                if (args.max_chars > 0 && chars > args.max_chars)
                {
                    text = truncate_utf8(text, args.max_chars);
                }

                std::size_t before = write_buffer.size();
                if (sig.bos_id >= 0)
                {
                    write_buffer.push_back(static_cast<std::uint32_t>(sig.bos_id));
                }
                tokenizer.encode_text_append(text, cache, write_buffer);
                if (sig.eos_id >= 0)
                {
                    write_buffer.push_back(static_cast<std::uint32_t>(sig.eos_id));
                }
                if (effective_cache_max_entries == 0 || cache.size() > effective_cache_max_entries)
                {
                    cache.clear();
                    cache.rehash(0);
                }

                ++result.num_docs;
                ++docs_pending_report;
                result.num_tokens += static_cast<std::uint64_t>(write_buffer.size() - before);
                if (docs_pending_report >= report_docs_batch && report_docs)
                {
                    report_docs(docs_pending_report);
                    docs_pending_report = 0;
                }
                if (write_buffer.size() >= flush_threshold_tokens && !flush_buffer())
                {
                    callback_ok = false;
                }
            },
            err);
        if (!read_ok)
        {
            if (err.empty())
            {
                err = "failed to read input file: " + task.path;
            }
            return false;
        }
        if (!callback_ok)
        {
            return false;
        }
        if (!flush_buffer())
        {
            return false;
        }
        if (docs_pending_report > 0 && report_docs)
        {
            report_docs(docs_pending_report);
        }

        result.reused = false;
        return true;
    }

    struct EncodeBatchIn
    {
        std::size_t id = 0;
        std::vector<std::string> docs;
    };
    struct EncodeBatchOut
    {
        std::size_t id = 0;
        std::vector<std::uint32_t> tokens;
        std::uint64_t num_docs = 0;
        std::uint64_t num_tokens = 0;
    };

    const std::size_t batch_docs = std::max<std::size_t>(args.encode_batch_size, 1);
    const std::size_t in_queue_cap = std::max<std::size_t>(encode_threads * 4, 8);
    constexpr std::uint64_t report_docs_batch = 128;

    std::atomic<bool> had_error{false};
    std::mutex err_mu;
    std::string shared_err;

    std::deque<EncodeBatchIn> in_queue;
    std::mutex in_mu;
    std::condition_variable in_cv;
    bool input_done = false;

    std::unordered_map<std::size_t, EncodeBatchOut> out_ready;
    std::mutex out_mu;
    std::condition_variable out_cv;
    std::size_t workers_done = 0;

    std::uint64_t skipped_docs = 0;
    std::mutex skipped_mu;

    auto set_error = [&](const std::string &message) {
        had_error.store(true, std::memory_order_relaxed);
        if (!message.empty())
        {
            std::lock_guard<std::mutex> lock(err_mu);
            if (shared_err.empty())
            {
                shared_err = message;
            }
        }
        in_cv.notify_all();
        out_cv.notify_all();
    };

    auto producer = [&]() {
        std::size_t next_batch_id = 0;
        EncodeBatchIn pending;
        pending.id = next_batch_id;
        pending.docs.reserve(batch_docs);
        std::uint64_t local_skipped = 0;

        auto push_pending = [&]() -> bool {
            if (pending.docs.empty())
            {
                return true;
            }
            std::unique_lock<std::mutex> lock(in_mu);
            in_cv.wait(lock, [&]() { return had_error.load(std::memory_order_relaxed) || in_queue.size() < in_queue_cap; });
            if (had_error.load(std::memory_order_relaxed))
            {
                return false;
            }
            in_queue.push_back(std::move(pending));
            pending = {};
            ++next_batch_id;
            pending.id = next_batch_id;
            pending.docs.reserve(batch_docs);
            lock.unlock();
            in_cv.notify_all();
            return true;
        };

        std::string local_err;
        bool callback_ok = true;
        bool read_ok = for_each_text_record(
            task.path, args.text_field,
            [&](const std::string &incoming_text) {
                if (!callback_ok || had_error.load(std::memory_order_relaxed))
                {
                    return;
                }
                if (incoming_text.empty())
                {
                    return;
                }
                std::string text = incoming_text;
                std::size_t chars = utf8_char_count(text);
                if (chars < args.min_chars)
                {
                    ++local_skipped;
                    return;
                }
                if (args.max_chars > 0 && chars > args.max_chars)
                {
                    text = truncate_utf8(text, args.max_chars);
                }
                pending.docs.push_back(std::move(text));
                if (pending.docs.size() >= batch_docs && !push_pending())
                {
                    callback_ok = false;
                }
            },
            local_err);

        if (read_ok && callback_ok && !had_error.load(std::memory_order_relaxed))
        {
            if (!push_pending())
            {
                callback_ok = false;
            }
        }

        {
            std::lock_guard<std::mutex> lock(skipped_mu);
            skipped_docs = local_skipped;
        }

        if (!read_ok)
        {
            set_error(local_err.empty() ? ("failed to read input file: " + task.path) : local_err);
        }
        else if (!callback_ok && !had_error.load(std::memory_order_relaxed))
        {
            set_error("failed to enqueue encode batches");
        }

        {
            std::lock_guard<std::mutex> lock(in_mu);
            input_done = true;
        }
        in_cv.notify_all();
    };

    auto worker = [&]() {
        std::unordered_map<std::string, std::vector<std::uint32_t>> local_cache;
        if (effective_cache_max_entries > 0)
        {
            const std::size_t reserve_n = std::min<std::size_t>(effective_cache_max_entries, 1 << 16);
            local_cache.reserve(reserve_n);
        }

        while (true)
        {
            EncodeBatchIn in_batch;
            {
                std::unique_lock<std::mutex> lock(in_mu);
                in_cv.wait(lock, [&]() {
                    return had_error.load(std::memory_order_relaxed) || !in_queue.empty() || input_done;
                });
                if (had_error.load(std::memory_order_relaxed) && in_queue.empty())
                {
                    break;
                }
                if (in_queue.empty())
                {
                    if (input_done)
                    {
                        break;
                    }
                    continue;
                }
                in_batch = std::move(in_queue.front());
                in_queue.pop_front();
            }
            in_cv.notify_all();

            EncodeBatchOut out_batch;
            out_batch.id = in_batch.id;
            for (const auto &text : in_batch.docs)
            {
                if (sig.bos_id >= 0)
                {
                    out_batch.tokens.push_back(static_cast<std::uint32_t>(sig.bos_id));
                }
                tokenizer.encode_text_append(text, local_cache, out_batch.tokens);
                if (sig.eos_id >= 0)
                {
                    out_batch.tokens.push_back(static_cast<std::uint32_t>(sig.eos_id));
                }
                ++out_batch.num_docs;
            }
            out_batch.num_tokens = static_cast<std::uint64_t>(out_batch.tokens.size());
            if (effective_cache_max_entries == 0 || local_cache.size() > effective_cache_max_entries)
            {
                local_cache.clear();
                local_cache.rehash(0);
            }

            {
                std::lock_guard<std::mutex> lock(out_mu);
                out_ready.emplace(out_batch.id, std::move(out_batch));
            }
            out_cv.notify_all();
        }

        {
            std::lock_guard<std::mutex> lock(out_mu);
            ++workers_done;
        }
        out_cv.notify_all();
    };

    std::thread producer_thread(producer);

    std::vector<std::thread> encode_workers;
    encode_workers.reserve(encode_threads);
    for (std::size_t i = 0; i < encode_threads; ++i)
    {
        encode_workers.emplace_back(worker);
    }

    std::size_t next_write_batch = 0;
    std::uint64_t docs_pending_report = 0;
    while (true)
    {
        EncodeBatchOut out_batch;
        bool has_batch = false;
        {
            std::unique_lock<std::mutex> lock(out_mu);
            out_cv.wait(lock, [&]() {
                return had_error.load(std::memory_order_relaxed) || out_ready.find(next_write_batch) != out_ready.end() ||
                       workers_done == encode_threads;
            });

            auto it = out_ready.find(next_write_batch);
            if (it != out_ready.end())
            {
                out_batch = std::move(it->second);
                out_ready.erase(it);
                has_batch = true;
            }
            else if (workers_done == encode_threads)
            {
                break;
            }
        }

        if (!has_batch)
        {
            if (had_error.load(std::memory_order_relaxed))
            {
                break;
            }
            continue;
        }

        if (!shard_writer.append_tokens(out_batch.tokens, err))
        {
            set_error(err);
            break;
        }
        result.num_docs += out_batch.num_docs;
        result.num_tokens += out_batch.num_tokens;
        docs_pending_report += out_batch.num_docs;
        if (docs_pending_report >= report_docs_batch && report_docs)
        {
            report_docs(docs_pending_report);
            docs_pending_report = 0;
        }
        ++next_write_batch;
    }

    producer_thread.join();
    for (auto &t : encode_workers)
    {
        t.join();
    }

    if (docs_pending_report > 0 && report_docs)
    {
        report_docs(docs_pending_report);
    }
    {
        std::lock_guard<std::mutex> lock(skipped_mu);
        result.num_skipped = skipped_docs;
    }
    if (had_error.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lock(err_mu);
        if (!shared_err.empty())
        {
            err = shared_err;
        }
        else if (err.empty())
        {
            err = "failed to process file: " + task.path;
        }
        return false;
    }

    result.reused = false;
    return true;
}

static bool remove_old_shards(const std::filesystem::path &out_dir)
{
    std::error_code ec;
    for (std::filesystem::directory_iterator it(out_dir, ec); !ec && it != std::filesystem::directory_iterator();
         it.increment(ec))
    {
        if (!it->is_regular_file())
        {
            continue;
        }
        std::string name = it->path().filename().string();
        if (starts_with(name, "train_") && ends_with(name, ".bin"))
        {
            std::filesystem::remove(it->path(), ec);
            if (ec)
            {
                return false;
            }
        }
    }
    return !ec;
}

static bool write_meta_json(const std::filesystem::path &meta_path, const Args &args, const std::string &data_glob,
                            const std::vector<std::string> &input_files, std::size_t vocab_size,
                            const std::string &dtype_name, std::int64_t eos_id, std::int64_t bos_id,
                            std::uint64_t num_docs, std::uint64_t num_skipped, std::uint64_t total_tokens,
                            const std::vector<ShardInfo> &shards, std::size_t reused_files, std::string &err)
{
    std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        err = "failed to write meta.json: " + meta_path.string();
        return false;
    }
    out << "{\n";
    out << "  \"created_at\": \"" << json_escape(now_string()) << "\",\n";
    out << "  \"tokenizer_path\": \"" << json_escape(args.tokenizer_path) << "\",\n";
    out << "  \"text_field\": \"" << json_escape(args.text_field) << "\",\n";
    out << "  \"data_glob\": \"" << json_escape(data_glob) << "\",\n";
    out << "  \"num_input_files\": " << input_files.size() << ",\n";
    out << "  \"input_files\": [\n";
    for (std::size_t i = 0; i < input_files.size(); ++i)
    {
        out << "    \"" << json_escape(input_files[i]) << "\"";
        if (i + 1 < input_files.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"dtype\": \"" << dtype_name << "\",\n";
    if (!args.eos_token.empty())
    {
        out << "  \"eos_token\": \"" << json_escape(args.eos_token) << "\",\n";
        out << "  \"eos_id\": " << eos_id << ",\n";
    }
    else
    {
        out << "  \"eos_token\": null,\n";
        out << "  \"eos_id\": null,\n";
    }
    if (!args.bos_token.empty())
    {
        out << "  \"bos_token\": \"" << json_escape(args.bos_token) << "\",\n";
        out << "  \"bos_id\": " << bos_id << ",\n";
    }
    else
    {
        out << "  \"bos_token\": null,\n";
        out << "  \"bos_id\": null,\n";
    }
    out << "  \"max_tokens_per_shard\": " << args.max_tokens_per_shard << ",\n";
    out << "  \"max_memory_mb\": " << args.max_memory_mb << ",\n";
    out << "  \"num_docs\": " << num_docs << ",\n";
    out << "  \"num_skipped\": " << num_skipped << ",\n";
    out << "  \"total_tokens\": " << total_tokens << ",\n";
    out << "  \"shards\": [\n";
    for (std::size_t i = 0; i < shards.size(); ++i)
    {
        out << "    {\"file\": \"" << json_escape(shards[i].file) << "\", \"num_tokens\": " << shards[i].num_tokens
            << "}";
        if (i + 1 < shards.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"num_reused_files\": " << reused_files << ",\n";
    out << "  \"layout\": {\"shards\": \"shards\", \"completed\": \"cache/completed.list\"}\n";
    out << "}\n";
    if (!out)
    {
        err = "failed to flush meta.json: " + meta_path.string();
        return false;
    }
    return true;
}

int run_tokenize(const Args &args)
{
    auto env = read_env_file(args.env_file);
    std::string data_glob = args.data_glob;
    if (data_glob.empty())
    {
        auto it = env.find("DATA_PATH");
        if (it != env.end())
        {
            data_glob = it->second;
        }
    }
    if (data_glob.empty())
    {
        std::cerr << "DATA_PATH is missing. Set it in .env or pass --data-glob.\n";
        return 1;
    }

    auto files = expand_data_files(data_glob);
    if (files.empty())
    {
        std::cerr << "No files matched: " << data_glob << "\n";
        return 1;
    }

    TokenizerEncoder tokenizer;
    std::string err;
    if (!tokenizer.load(args.tokenizer_path, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    std::uint32_t eos_id_u32 = 0;
    std::int64_t eos_id = -1;
    if (!args.eos_token.empty())
    {
        if (!tokenizer.token_to_id(args.eos_token, eos_id_u32))
        {
            std::cerr << "eos token not found in tokenizer: " << args.eos_token << "\n";
            return 1;
        }
        eos_id = static_cast<std::int64_t>(eos_id_u32);
    }

    std::uint32_t bos_id_u32 = 0;
    std::int64_t bos_id = -1;
    if (!args.bos_token.empty())
    {
        if (!tokenizer.token_to_id(args.bos_token, bos_id_u32))
        {
            std::cerr << "bos token not found in tokenizer: " << args.bos_token << "\n";
            return 1;
        }
        bos_id = static_cast<std::int64_t>(bos_id_u32);
    }

    std::uint32_t dtype_bytes = tokenizer.vocab_size() <= std::numeric_limits<std::uint16_t>::max() ? 2u : 4u;
    std::string dtype_name = dtype_bytes == 2 ? "uint16" : "uint32";

    std::filesystem::path out_root = args.out_dir;
    std::filesystem::path shard_dir = out_root / "shards";
    std::filesystem::path cache_dir = out_root / "cache";
    std::filesystem::path completed_list_path = cache_dir / "completed.list";
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    if (ec)
    {
        std::cerr << "failed to create cache dir under: " << out_root.string() << "\n";
        return 1;
    }
    ec.clear();
    std::filesystem::create_directories(shard_dir, ec);
    if (ec)
    {
        std::cerr << "failed to create shard dir under: " << out_root.string() << "\n";
        return 1;
    }

    std::string tokenizer_fp = make_tokenizer_fingerprint(args.tokenizer_path);
    if (tokenizer_fp.empty())
    {
        std::cerr << "failed to fingerprint tokenizer file: " << args.tokenizer_path << "\n";
        return 1;
    }

    std::vector<FileTask> tasks;
    tasks.reserve(files.size());
    for (std::size_t i = 0; i < files.size(); ++i)
    {
        std::uint64_t sz = 0;
        std::int64_t mt = 0;
        if (!get_file_stat(files[i], sz, mt))
        {
            std::cerr << "failed to stat input file: " << files[i] << "\n";
            return 1;
        }
        FileTask t;
        t.index = i;
        t.path = files[i];
        t.normalized_path = normalize_path_for_compare(files[i]);
        t.file_size = sz;
        t.file_mtime = mt;
        tasks.push_back(std::move(t));
    }

    PartSignature sig;
    sig.tokenizer_fingerprint = tokenizer_fp;
    sig.text_field = args.text_field;
    sig.min_chars = args.min_chars;
    sig.max_chars = args.max_chars;
    sig.bos_id = bos_id;
    sig.eos_id = eos_id;
    sig.dtype_bytes = dtype_bytes;
    std::string run_signature = make_run_signature(sig);

    std::unordered_map<std::string, CompletionRecord> completion_records;
    if (!init_completion_records(completed_list_path, run_signature, args.resume, completion_records, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    bool resume_existing_shards = args.resume && !completion_records.empty();
    ShardWriter shard_writer(shard_dir, dtype_bytes, args.max_tokens_per_shard);
    if (!shard_writer.initialize(resume_existing_shards, err))
    {
        std::cerr << err << "\n";
        return 1;
    }
    if (resume_existing_shards)
    {
        std::uint64_t completed_tokens = 0;
        for (const auto &kv : completion_records)
        {
            completed_tokens += kv.second.result.num_tokens;
        }
        if (completed_tokens != shard_writer.total_tokens())
        {
            std::cerr << "resume state mismatch: completed.list tokens=" << completed_tokens
                      << ", existing shards tokens=" << shard_writer.total_tokens()
                      << ". Please run with --no-resume or clean output dir.\n";
            return 1;
        }
    }

    std::size_t worker_threads = args.threads;
    if (worker_threads == 0)
    {
        worker_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    bool single_large_file_mode = tasks.size() == 1 && worker_threads > 1;
    std::size_t file_worker_threads = single_large_file_mode ? 1 : worker_threads;
    std::size_t per_file_encode_threads = single_large_file_mode ? worker_threads : 1;

    std::uint64_t total_docs_est = 0;
    if (args.prescan_records)
    {
        err.clear();
        if (!prescan_total_docs(tasks, args, total_docs_est, err))
        {
            std::cerr << err << "\n";
            return 1;
        }
    }

    std::cerr << "Files: " << files.size() << "\n";
    std::cerr << "Threads(file-level): " << file_worker_threads << "\n";
    if (per_file_encode_threads > 1)
    {
        std::cerr << "Threads(in-file encode): " << per_file_encode_threads << "\n";
    }
    std::cerr << "Token piece cache entries/worker: " << args.cache_max_entries << "\n";
    if (args.max_memory_mb > 0)
    {
        std::cerr << "Memory cap/worker: " << args.max_memory_mb << " MiB\n";
    }
    std::cerr << "Prescan docs: " << (args.prescan_records ? "on" : "off") << "\n";
    if (args.prescan_records && total_docs_est > 0)
    {
        std::cerr << "Total docs (estimated): " << total_docs_est << "\n";
    }
    std::cerr << "Tokenizer model: " << tokenizer.model_name() << "\n";
    std::cerr << "Tokenizer vocab: " << tokenizer.vocab_size() << " (dtype=" << dtype_name << ")\n";
    std::cerr << "Output root: " << out_root.string() << "\n";
    std::cerr << "Shard dir: " << shard_dir.string() << "\n";
    std::cerr << "Completed list: " << completed_list_path.string() << "\n";
    std::cerr << "Resume completed files: " << completion_records.size() << "\n";

    auto start_time = std::chrono::steady_clock::now();
    std::vector<PartResult> results(tasks.size());
    std::atomic<std::size_t> next_idx{0};
    std::atomic<bool> had_error{false};
    std::mutex err_mu;
    std::mutex completion_mu;
    std::string shared_err;
    ProgressTracker progress(tasks.size(), "tokenizing", 1000);
    if (args.prescan_records && total_docs_est > 0)
    {
        progress.set_total_docs(total_docs_est);
    }

    auto worker = [&]() {
        auto report_docs = [&](std::uint64_t docs) {
            if (docs > 0)
            {
                progress.add(0, docs);
            }
        };
        while (true)
        {
            if (had_error.load(std::memory_order_relaxed))
            {
                break;
            }
            std::size_t idx = next_idx.fetch_add(1);
            if (idx >= tasks.size())
            {
                break;
            }
            PartResult r;
            std::string local_err;
            if (args.resume && completion_is_reusable(tasks[idx], completion_records, r))
            {
                results[idx] = r;
                progress.add(1, r.num_docs);
                continue;
            }
            if (!process_file_to_shards(tasks[idx], args, tokenizer, sig, shard_writer, per_file_encode_threads, r,
                                        report_docs, local_err))
            {
                had_error.store(true);
                std::lock_guard<std::mutex> lock(err_mu);
                if (shared_err.empty())
                {
                    shared_err = local_err;
                }
                continue;
            }
            if (!append_completion_record(completed_list_path, tasks[idx], r, completion_mu, local_err))
            {
                had_error.store(true);
                std::lock_guard<std::mutex> lock(err_mu);
                if (shared_err.empty())
                {
                    shared_err = local_err;
                }
                continue;
            }
            results[idx] = r;
            progress.add(1, 0);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(file_worker_threads);
    for (std::size_t t = 0; t < file_worker_threads; ++t)
    {
        threads.emplace_back(worker);
    }
    for (auto &t : threads)
    {
        t.join();
    }
    progress.finish();
    if (had_error.load())
    {
        std::cerr << (shared_err.empty() ? "tokenization failed" : shared_err) << "\n";
        return 1;
    }

    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::uint64_t expected_tokens = 0;
    std::size_t reused_files = 0;
    for (const auto &r : results)
    {
        num_docs += r.num_docs;
        num_skipped += r.num_skipped;
        expected_tokens += r.num_tokens;
        if (r.reused)
        {
            ++reused_files;
        }
    }

    std::vector<ShardInfo> shards;
    std::uint64_t total_tokens = 0;
    err.clear();
    if (!shard_writer.finalize(shards, total_tokens, err))
    {
        std::cerr << err << "\n";
        return 1;
    }
    if (expected_tokens != total_tokens)
    {
        std::cerr << "token count mismatch: expected " << expected_tokens << " from completion records, got "
                  << total_tokens << " in shards\n";
        return 1;
    }

    std::filesystem::path meta_path = out_root / "meta.json";
    if (!write_meta_json(meta_path, args, data_glob, files, tokenizer.vocab_size(), dtype_name, eos_id, bos_id, num_docs,
                         num_skipped, total_tokens, shards, reused_files, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time)
            .count();
    if (elapsed < 1e-9)
    {
        elapsed = 1e-9;
    }
    std::cerr << "done. docs=" << num_docs << " skipped=" << num_skipped << " total_tokens=" << total_tokens << "\n";
    std::cerr << "shards=" << shards.size() << " dtype=" << dtype_name << " out=" << shard_dir.string() << "\n";
    std::cerr << "reused_files=" << reused_files << "/" << files.size() << "\n";
    std::cerr << "throughput docs/s=" << static_cast<double>(num_docs) / elapsed
              << " tok/s=" << static_cast<double>(total_tokens) / elapsed << "\n";
    return 0;
}

} // namespace tokenflux::tokenize
