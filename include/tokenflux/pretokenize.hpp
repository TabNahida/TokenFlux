#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tokenflux/tokenizer.hpp"

namespace tokenflux {

struct PretokenizeOptions {
  std::size_t chunk_bytes = 1 << 20;
  std::size_t num_threads = 0;
  bool mmap_output = false;
};

struct PretokenizeRecord {
  std::uint64_t offset;
  std::uint32_t length;
};

class PretokenizeEngine {
 public:
  explicit PretokenizeEngine(const Tokenizer& tokenizer);

  void PretokenizeFiles(const std::vector<std::string>& input_files,
                        const std::string& output_bin,
                        const std::string& output_index,
                        PretokenizeOptions options = {}) const;

 private:
  const Tokenizer& tokenizer_;
};

}  // namespace tokenflux

