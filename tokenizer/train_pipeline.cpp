#include "train_pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include "input_source.hpp"
#include "train_frontend.hpp"
#include "train_io.hpp"
#include "trainers.hpp"

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
    return static_cast<std::size_t>((total_budget_bytes * 3ull / 4ull) / 128ull);
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

int run_train(Config cfg)
{
    if (cfg.data_glob.empty() && cfg.data_list.empty())
    {
        std::cerr << "DATA_PATH / DATA_LIST not set and no input was provided.\n";
        return 1;
    }

    std::vector<InputSource> sources;
    std::string err;
    std::string remote_cache_dir = (std::filesystem::path(cfg.chunk_dir) / "remote_inputs").string();
    if (!resolve_input_sources(cfg.data_glob, cfg.data_list, remote_cache_dir, sources, err))
    {
        std::cerr << err << "\n";
        return 1;
    }
    if (sources.empty())
    {
        std::cerr << "No files matched input settings.\n";
        return 1;
    }

    std::vector<std::string> files;
    files.reserve(sources.size());
    for (const auto &source : sources)
    {
        files.push_back(source.local_path);
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
    std::cerr << "Input mode: " << (cfg.data_list.empty() ? "glob" : "list") << "\n";
    std::cerr << "Streaming prescan: " << (cfg.prescan_records ? "on" : "off") << "\n";
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
    if (!build_count_chunks(cfg, sources, local_entry_cap, process_text, chunk_stats, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    GlobalCountMap global_counts;
    std::uint64_t total_docs = 0;
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
