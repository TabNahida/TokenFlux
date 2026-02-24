#pragma once

#include "tokenflux/corpus_reader.hpp"
#include "tokenflux/tokenizer.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace tokenflux {

struct PretokenizeOptions {
    size_t threads = 0;
    size_t batch_size = 4096;
    std::string output_dir = "./pretokenized";
};

class PretokenizePipeline {
public:
    PretokenizePipeline(const Tokenizer& tokenizer, CorpusReadOptions ropts = {});

    bool run(const std::vector<std::string>& files, const PretokenizeOptions& options) const;

private:
    const Tokenizer& tokenizer_;
    CorpusReader reader_;
};

} // namespace tokenflux
