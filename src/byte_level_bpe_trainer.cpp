#include "tokenflux/byte_level_bpe_trainer.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <thread>
#include <unordered_map>

namespace tokenflux {

ByteLevelBPETrainer::ByteLevelBPETrainer(TrainerOptions options, CorpusReadOptions ropts)
    : options_(std::move(options)), reader_(std::move(ropts)) {}

std::vector<ByteLevelBPETrainer::WordItem> ByteLevelBPETrainer::build_initial_corpus(const std::vector<std::string>& files) const {
    std::unordered_map<std::string, uint32_t> counts;
    for (const auto& f : files) {
        reader_.for_each_record(f, [&](const std::string& line) {
            std::string cur;
            for (char c : line) {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    if (!cur.empty()) {
                        ++counts[cur];
                        cur.clear();
                    }
                } else {
                    cur.push_back(c);
                }
            }
            if (!cur.empty()) ++counts[cur];
        });
    }

    std::vector<WordItem> words;
    words.reserve(counts.size());
    for (auto& [w, f] : counts) {
        WordItem item;
        item.freq = f;
        item.symbols.reserve(w.size());
        for (unsigned char ch : w) item.symbols.push_back(static_cast<int32_t>(ch));
        words.push_back(std::move(item));
    }
    return words;
}

std::pair<int32_t, int32_t> ByteLevelBPETrainer::find_best_pair(const std::vector<WordItem>& words) const {
    size_t threads = options_.threads > 0 ? options_.threads : std::max(1u, std::thread::hardware_concurrency());
    size_t per_chunk = std::max<size_t>(1, words.size() / threads + 1);

    using Pair = uint64_t;
    auto pack = [](int32_t a, int32_t b) -> Pair {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    };

    std::vector<std::future<std::unordered_map<Pair, uint64_t>>> jobs;
    for (size_t t = 0; t < threads; ++t) {
        size_t start = t * per_chunk;
        if (start >= words.size()) break;
        size_t end = std::min(words.size(), start + per_chunk);
        jobs.emplace_back(std::async(std::launch::async, [&, start, end]() {
            std::unordered_map<Pair, uint64_t> local;
            local.reserve(options_.topk_pairs);
            for (size_t i = start; i < end; ++i) {
                const auto& syms = words[i].symbols;
                for (size_t j = 0; j + 1 < syms.size(); ++j) {
                    auto key = pack(syms[j], syms[j + 1]);
                    local[key] += words[i].freq;
                }
                if (local.size() > options_.topk_pairs * 2) {
                    std::vector<std::pair<Pair, uint64_t>> temp(local.begin(), local.end());
                    std::nth_element(temp.begin(), temp.begin() + static_cast<std::ptrdiff_t>(options_.topk_pairs), temp.end(),
                                     [](const auto& l, const auto& r) { return l.second > r.second; });
                    local.clear();
                    for (size_t k = 0; k < options_.topk_pairs; ++k) local[temp[k].first] = temp[k].second;
                }
            }
            return local;
        }));
    }

    std::unordered_map<Pair, uint64_t> merged;
    for (auto& job : jobs) {
        auto local = job.get();
        for (auto& [k, v] : local) merged[k] += v;
    }

    Pair best_key = 0;
    uint64_t best_freq = 0;
    for (auto& [k, v] : merged) {
        if (v > best_freq) {
            best_freq = v;
            best_key = k;
        }
    }

    if (best_freq == 0) return {-1, -1};
    return {static_cast<int32_t>(best_key >> 32), static_cast<int32_t>(best_key & 0xffffffff)};
}

void ByteLevelBPETrainer::merge_pair(std::vector<WordItem>& words, std::pair<int32_t, int32_t> pair, int32_t new_id) const {
    size_t threads = options_.threads > 0 ? options_.threads : std::max(1u, std::thread::hardware_concurrency());
    size_t per_chunk = std::max<size_t>(1, words.size() / threads + 1);

    std::vector<std::future<void>> jobs;
    for (size_t t = 0; t < threads; ++t) {
        size_t start = t * per_chunk;
        if (start >= words.size()) break;
        size_t end = std::min(words.size(), start + per_chunk);
        jobs.emplace_back(std::async(std::launch::async, [&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                auto& syms = words[i].symbols;
                std::vector<int32_t> out;
                out.reserve(syms.size());
                for (size_t j = 0; j < syms.size();) {
                    if (j + 1 < syms.size() && syms[j] == pair.first && syms[j + 1] == pair.second) {
                        out.push_back(new_id);
                        j += 2;
                    } else {
                        out.push_back(syms[j]);
                        ++j;
                    }
                }
                syms.swap(out);
            }
        }));
    }
    for (auto& j : jobs) j.get();
}

Tokenizer ByteLevelBPETrainer::train(const std::vector<std::string>& files) const {
    auto words = build_initial_corpus(files);

    std::vector<std::string> vocab;
    vocab.reserve(options_.vocab_size);
    for (int i = 0; i < 256; ++i) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
        vocab.push_back(buf);
    }

    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(options_.vocab_size > 256 ? options_.vocab_size - 256 : 0);

    for (size_t iter = 0; iter < options_.max_iterations && vocab.size() < options_.vocab_size; ++iter) {
        auto best = find_best_pair(words);
        if (best.first < 0) break;

        auto a = vocab[static_cast<size_t>(best.first)];
        auto b = vocab[static_cast<size_t>(best.second)];
        auto merged = a + b;

        int32_t new_id = static_cast<int32_t>(vocab.size());
        vocab.push_back(merged);
        merges.push_back({a, b});
        merge_pair(words, best, new_id);
    }

    Tokenizer tok;
    tok.set_vocab(std::move(vocab));
    tok.set_merges(std::move(merges));
    return tok;
}

} // namespace tokenflux
