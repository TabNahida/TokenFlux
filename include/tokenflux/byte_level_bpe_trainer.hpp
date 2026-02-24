#pragma once

#include "tokenflux/corpus_reader.hpp"
#include "tokenflux/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tokenflux {

struct TrainerOptions {
    size_t vocab_size = 32000;
    size_t max_iterations = 1000;
    size_t threads = 0;
    size_t chunk_size = 1 << 16;
    size_t topk_pairs = 200000;
    size_t memory_limit_mb = 4096;
};

class ByteLevelBPETrainer {
public:
    explicit ByteLevelBPETrainer(TrainerOptions options = {}, CorpusReadOptions ropts = {});

    Tokenizer train(const std::vector<std::string>& files) const;

private:
    struct WordItem {
        std::vector<int32_t> symbols;
        uint32_t freq = 1;
    };

    std::vector<WordItem> build_initial_corpus(const std::vector<std::string>& files) const;
    std::pair<int32_t, int32_t> find_best_pair(const std::vector<WordItem>& words) const;
    void merge_pair(std::vector<WordItem>& words, std::pair<int32_t, int32_t> pair, int32_t new_id) const;

    TrainerOptions options_;
    CorpusReader reader_;
};

} // namespace tokenflux
