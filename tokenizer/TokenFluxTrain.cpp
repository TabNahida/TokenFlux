#include "tokenflux_lib.h"
#include "train_frontend.h"
#include "train_io.h"
#include "trainers.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <thread>

namespace
{
std::size_t derive_local_entry_cap(const Config &cfg)
{
    std::size_t local_entry_cap = cfg.top_k;
    if (cfg.max_memory_mb == 0)
    {
        return local_entry_cap;
    }

    std::uint64_t total_budget_bytes = static_cast<std::uint64_t>(cfg.max_memory_mb) * 1024ull * 1024ull;
    std::uint64_t per_thread_budget = total_budget_bytes / std::max<std::size_t>(1, cfg.threads);
    std::size_t derived_local_cap = static_cast<std::size_t>(per_thread_budget / 160ull);
    if (derived_local_cap > 0)
    {
        if (local_entry_cap == 0 || local_entry_cap > derived_local_cap)
        {
            local_entry_cap = derived_local_cap;
        }
    }
    return local_entry_cap;
}

std::size_t derive_global_entry_cap(const Config &cfg)
{
    if (cfg.max_memory_mb == 0)
    {
        return 0;
    }
    std::uint64_t total_budget_bytes = static_cast<std::uint64_t>(cfg.max_memory_mb) * 1024ull * 1024ull;
    std::size_t derived_global_cap = static_cast<std::size_t>((total_budget_bytes * 3ull / 4ull) / 128ull);
    return derived_global_cap;
}

void derive_pair_cap(Config &cfg)
{
    if (cfg.max_memory_mb == 0 || cfg.pair_max_entries > 0)
    {
        return;
    }
    std::uint64_t total_budget_bytes = static_cast<std::uint64_t>(cfg.max_memory_mb) * 1024ull * 1024ull;
    cfg.pair_max_entries = static_cast<std::size_t>((total_budget_bytes / 2ull) / 40ull);
    cfg.pair_max_entries = std::max<std::size_t>(cfg.pair_max_entries, 100'000);
}
} // namespace

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Config cfg;

    cfg.env_path = detect_env_path_arg(argc, argv, cfg.env_path);
    auto env = read_env_file(cfg.env_path);
    apply_env_overrides(cfg, env);

    std::string parse_err;
    bool show_help = false;
    if (!parse_train_args(argc, argv, cfg, parse_err, show_help))
    {
        if (show_help)
        {
            print_train_usage();
            return 0;
        }
        std::cerr << parse_err << "\n";
        print_train_usage();
        return 1;
    }

    if (cfg.data_glob.empty())
    {
        std::cerr << "DATA_PATH not set in .env and --data not provided.\n";
        return 1;
    }

    auto files = expand_data_glob(cfg.data_glob);
    if (files.empty())
    {
        std::cerr << "No files matched: " << cfg.data_glob << "\n";
        return 1;
    }

    if (cfg.threads == 0)
    {
        cfg.threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    if (cfg.records_per_chunk == 0)
    {
        cfg.records_per_chunk = 1;
    }

    std::size_t local_entry_cap = derive_local_entry_cap(cfg);
    std::size_t global_entry_cap = derive_global_entry_cap(cfg);
    derive_pair_cap(cfg);

    std::error_code ec;
    std::filesystem::create_directories(cfg.chunk_dir, ec);
    if (ec)
    {
        std::cerr << "Failed to create chunk dir: " << cfg.chunk_dir << "\n";
        return 1;
    }

    std::cerr << "Trainer: " << trainer_kind_to_string(cfg.trainer) << "\n";
    std::cerr << "Files: " << files.size() << "\n";
    std::cerr << "Threads: " << cfg.threads << "\n";
    std::cerr << "Records/chunk: " << cfg.records_per_chunk << "\n";
    std::cerr << "Chunk dir: " << cfg.chunk_dir << "\n";
    if (local_entry_cap > 0)
    {
        std::cerr << "Local count cap/chunk: " << local_entry_cap << "\n";
    }
    if (global_entry_cap > 0)
    {
        std::cerr << "Global count cap: " << global_entry_cap << "\n";
    }
    if (cfg.pair_max_entries > 0)
    {
        std::cerr << "Pair cap entries: " << cfg.pair_max_entries << "\n";
    }

    auto process_text = build_process_text_callback(cfg);
    ChunkBuildStats chunk_stats;
    std::string err;
    if (!build_count_chunks(cfg, files, local_entry_cap, process_text, chunk_stats, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    GlobalCountMap global_counts;
    uint64_t total_docs = 0;
    if (!merge_count_chunks(cfg, chunk_stats.total_chunks, global_entry_cap, global_counts, total_docs, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    std::cerr << "Docs: " << total_docs << "\n";
    std::cerr << "Unique words after chunk merge: " << global_counts.size() << "\n";

    TrainArtifacts artifacts;
    if (!train_from_global_counts(cfg, global_counts, artifacts, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    if (!write_trained_tokenizer(cfg, artifacts, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    std::cerr << "Saved tokenizer: " << cfg.output_json << "\n";
    if (cfg.write_vocab)
    {
        std::cerr << "Saved vocab: " << cfg.output_vocab << "\n";
    }
    if (cfg.write_merges && artifacts.has_merges)
    {
        std::cerr << "Saved merges: " << cfg.output_merges << "\n";
    }

    return 0;
}

