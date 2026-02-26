#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenflux_config.h"

using LocalCountMap = std::unordered_map<std::string, uint32_t>;
using GlobalCountMap = std::unordered_map<std::string, uint64_t>;
using ProcessTextFn =
    std::function<void(const std::string &, LocalCountMap &, uint64_t &, std::size_t &, std::size_t local_entry_cap)>;

struct ChunkBuildStats
{
    std::size_t total_chunks = 0;
    uint64_t total_docs = 0;
};

std::string chunk_path_for_id(const Config &cfg, std::size_t chunk_id);

bool build_count_chunks(const Config &cfg, const std::vector<std::string> &files, std::size_t local_entry_cap,
                        const ProcessTextFn &process_text, ChunkBuildStats &stats, std::string &err);

bool merge_count_chunks(const Config &cfg, std::size_t total_chunks, std::size_t global_entry_cap,
                        GlobalCountMap &global_counts, uint64_t &total_docs, std::string &err);
