#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tokenflux/tokenizer.hpp"

namespace tokenflux {

struct TrainerOptions {
  std::size_t vocab_size = 32000;
  std::size_t min_frequency = 2;
  std::size_t top_k_pairs = 200000;
  std::size_t chunk_size = 1 << 20;
  std::size_t max_memory_bytes = 2ull << 30;
  std::size_t num_threads = 0;
  bool byte_fallback = true;
};

struct TrainResult {
  Vocab vocab;
  MergeRules merges;
};

class ByteLevelBPETrainer {
 public:
  explicit ByteLevelBPETrainer(TrainerOptions options = {});

  [[nodiscard]] TrainResult TrainFromFiles(const std::vector<std::string>& files) const;
  [[nodiscard]] TrainResult TrainFromLines(const std::vector<std::string>& lines) const;

 private:
  struct PairHash {
    std::size_t operator()(const std::pair<std::uint32_t, std::uint32_t>& p) const noexcept;
  };

  [[nodiscard]] std::vector<std::vector<std::uint32_t>> PretokenizeLines(
      const std::vector<std::string>& lines) const;

  void CountPairsChunk(
      const std::vector<std::vector<std::uint32_t>>& corpus,
      std::size_t begin,
      std::size_t end,
      std::vector<std::pair<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>>& out) const;

  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> SelectBestPair(
      const std::vector<std::vector<std::uint32_t>>& corpus) const;

  void MergePair(std::vector<std::vector<std::uint32_t>>& corpus,
                 std::pair<std::uint32_t, std::uint32_t> pair,
                 std::uint32_t new_id) const;

  TrainerOptions options_;
};

}  // namespace tokenflux

