#include "tokenflux/pretokenizer.hpp"

#include <filesystem>
#include <fstream>
#include <future>
#include <thread>

namespace tokenflux {

PretokenizePipeline::PretokenizePipeline(const Tokenizer& tokenizer, CorpusReadOptions ropts)
    : tokenizer_(tokenizer), reader_(std::move(ropts)) {}

bool PretokenizePipeline::run(const std::vector<std::string>& files, const PretokenizeOptions& options) const {
    std::filesystem::create_directories(options.output_dir);
    size_t threads = options.threads > 0 ? options.threads : std::max(1u, std::thread::hardware_concurrency());

    std::atomic<size_t> file_idx{0};
    std::vector<std::future<bool>> jobs;
    for (size_t t = 0; t < threads; ++t) {
        jobs.emplace_back(std::async(std::launch::async, [&]() {
            while (true) {
                auto idx = file_idx.fetch_add(1);
                if (idx >= files.size()) break;
                const auto& file = files[idx];
                std::ofstream out(options.output_dir + "/shard_" + std::to_string(idx) + ".tokbin", std::ios::binary);
                if (!out) return false;
                bool ok = reader_.for_each_record(file, [&](const std::string& line) {
                    auto ids = tokenizer_.tokenize(line);
                    uint32_t n = static_cast<uint32_t>(ids.size());
                    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
                    out.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(int32_t)));
                });
                if (!ok) return false;
            }
            return true;
        }));
    }

    for (auto& j : jobs) {
        if (!j.get()) return false;
    }
    return true;
}

} // namespace tokenflux
