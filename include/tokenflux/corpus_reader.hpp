#pragma once

#include <functional>
#include <string>
#include <vector>

namespace tokenflux {

struct CorpusReadOptions {
    std::vector<std::string> json_text_fields = {"text", "content"};
};

class CorpusReader {
public:
    explicit CorpusReader(CorpusReadOptions options = {});

    bool for_each_record(const std::string& path, const std::function<void(const std::string&)>& fn) const;

private:
    bool read_text_like(const std::string& path, const std::function<void(const std::string&)>& fn) const;
    bool read_jsonl(const std::string& path, const std::function<void(const std::string&)>& fn) const;
    bool read_json(const std::string& path, const std::function<void(const std::string&)>& fn) const;
    bool read_json_gz(const std::string& path, const std::function<void(const std::string&)>& fn) const;
    bool read_parquet(const std::string& path, const std::function<void(const std::string&)>& fn) const;

    CorpusReadOptions options_;
};

} // namespace tokenflux
