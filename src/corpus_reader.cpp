#include "tokenflux/corpus_reader.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <zlib.h>

namespace tokenflux {

CorpusReader::CorpusReader(CorpusReadOptions options) : options_(std::move(options)) {}

bool CorpusReader::for_each_record(const std::string& path, const std::function<void(const std::string&)>& fn) const {
    auto ext = std::filesystem::path(path).extension().string();
    if (ext == ".txt") return read_text_like(path, fn);
    if (ext == ".jsonl") return read_jsonl(path, fn);
    if (ext == ".json") return read_json(path, fn);
    if (ext == ".gz") return read_json_gz(path, fn);
    if (ext == ".parquet") return read_parquet(path, fn);
    return read_text_like(path, fn);
}

bool CorpusReader::read_text_like(const std::string& path, const std::function<void(const std::string&)>& fn) const {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) fn(line);
    }
    return true;
}

bool CorpusReader::read_jsonl(const std::string& path, const std::function<void(const std::string&)>& fn) const {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, false);
        if (j.is_discarded()) continue;
        for (const auto& field : options_.json_text_fields) {
            if (j.contains(field) && j[field].is_string()) {
                fn(j[field].get<std::string>());
                break;
            }
        }
    }
    return true;
}

bool CorpusReader::read_json(const std::string& path, const std::function<void(const std::string&)>& fn) const {
    std::ifstream in(path);
    if (!in) return false;
    nlohmann::json j;
    in >> j;
    if (j.is_array()) {
        for (const auto& item : j) {
            for (const auto& field : options_.json_text_fields) {
                if (item.contains(field) && item[field].is_string()) {
                    fn(item[field].get<std::string>());
                    break;
                }
            }
        }
    } else if (j.is_object()) {
        for (const auto& field : options_.json_text_fields) {
            if (j.contains(field) && j[field].is_string()) {
                fn(j[field].get<std::string>());
                break;
            }
        }
    }
    return true;
}

bool CorpusReader::read_json_gz(const std::string& path, const std::function<void(const std::string&)>& fn) const {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) return false;
    std::string payload;
    char buf[1 << 15];
    int read_n = 0;
    while ((read_n = gzread(gz, buf, sizeof(buf))) > 0) {
        payload.append(buf, static_cast<size_t>(read_n));
    }
    gzclose(gz);

    auto j = nlohmann::json::parse(payload, nullptr, false);
    if (j.is_discarded()) {
        std::istringstream iss(payload);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            auto jl = nlohmann::json::parse(line, nullptr, false);
            if (jl.is_discarded()) continue;
            for (const auto& field : options_.json_text_fields) {
                if (jl.contains(field) && jl[field].is_string()) {
                    fn(jl[field].get<std::string>());
                    break;
                }
            }
        }
        return true;
    }

    if (j.is_array()) {
        for (const auto& item : j) {
            for (const auto& field : options_.json_text_fields) {
                if (item.contains(field) && item[field].is_string()) {
                    fn(item[field].get<std::string>());
                    break;
                }
            }
        }
    }
    return true;
}

bool CorpusReader::read_parquet(const std::string& path, const std::function<void(const std::string&)>& fn) const {
#ifdef TOKENFLUX_WITH_PARQUET
    (void)path;
    (void)fn;
    return false;
#else
    (void)path;
    (void)fn;
    return false;
#endif
}

} // namespace tokenflux
