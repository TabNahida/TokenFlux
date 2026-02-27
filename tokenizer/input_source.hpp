#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct InputSource
{
    std::string source;
    std::string local_path;
    std::string normalized_id;
    std::uint64_t file_size = 0;
    std::int64_t file_mtime = 0;
    bool remote = false;
};

bool is_remote_http_url(const std::string &value);
bool is_file_url(const std::string &value);
std::string normalize_input_id(const std::string &value);

bool resolve_input_sources(const std::vector<std::string> &input_entries, const std::string &data_glob,
                           const std::string &data_list,
                           const std::string &remote_cache_dir, std::vector<InputSource> &sources, std::string &err);
