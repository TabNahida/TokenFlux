#include "tokenflux/trainer.hpp"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace tokenflux {

namespace {
std::size_t EffectiveThreads(std::size_t configured) {
  if (configured > 0) {
    return configured;
  }
  const auto hw = std::thread::hardware_concurrency();
  return hw == 0 ? 4 : hw;
}

inline std::uint32_t CompressPair(std::uint32_t a, std::uint32_t b) {
  return (a * 1315423911u) ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2));
}

}  // namespace

ByteLevelBPETrainer::ByteLevelBPETrainer(TrainerOptions options) : options_(options) {}

std::size_t ByteLevelBPETrainer::PairHash::operator()(
    const std::pair<std::uint32_t, std::uint32_t>& p) const noexcept {
  return (static_cast<std::size_t>(p.first) << 32u) ^ p.second;
}

std::vector<std::vector<std::uint32_t>> ByteLevelBPETrainer::PretokenizeLines(
    const std::vector<std::string>& lines) const {
  std::vector<std::vector<std::uint32_t>> out(lines.size());

  for (std::size_t idx = 0; idx < lines.size(); ++idx) {
    const auto& line = lines[idx];
    auto& encoded = out[idx];
    encoded.reserve(line.size());
    for (unsigned char c : line) {
      encoded.push_back(c);
    }
  }

  return out;
}

void ByteLevelBPETrainer::CountPairsChunk(
    const std::vector<std::vector<std::uint32_t>>& corpus,
    std::size_t begin,
    std::size_t end,
    std::vector<std::pair<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>>& out) const {
  std::unordered_map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t, PairHash> counts;

  const std::size_t reserve_by_memory = options_.max_memory_bytes / 24;
  counts.reserve(std::min<std::size_t>(reserve_by_memory, options_.top_k_pairs * 2));

  for (std::size_t i = begin; i < end; ++i) {
    const auto& seq = corpus[i];
    for (std::size_t j = 0; j + 1 < seq.size(); ++j) {
      auto pair = std::make_pair(seq[j], seq[j + 1]);
      auto& v = counts[pair];
      if (v != std::numeric_limits<std::uint32_t>::max()) {
        ++v;
      }
    }
  }

  out.reserve(counts.size());
  for (const auto& kv : counts) {
    out.emplace_back(kv.first, kv.second);
  }

  if (out.size() > options_.top_k_pairs) {
    std::nth_element(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(options_.top_k_pairs), out.end(),
                     [](const auto& l, const auto& r) { return l.second > r.second; });
    out.resize(options_.top_k_pairs);
  }
}

std::pair<std::uint32_t, std::uint32_t> ByteLevelBPETrainer::SelectBestPair(
    const std::vector<std::vector<std::uint32_t>>& corpus) const {
  const std::size_t threads = EffectiveThreads(options_.num_threads);
  const std::size_t shard = std::max<std::size_t>(1, corpus.size() / threads);

  std::vector<std::future<std::vector<std::pair<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>>>> futures;
  futures.reserve(threads);

  for (std::size_t start = 0; start < corpus.size(); start += shard) {
    const std::size_t end = std::min(start + shard, corpus.size());
    futures.push_back(std::async(std::launch::async, [&, start, end] {
      std::vector<std::pair<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>> partial;
      CountPairsChunk(corpus, start, end, partial);
      return partial;
    }));
  }

  std::unordered_map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t, PairHash> merged;
  for (auto& future : futures) {
    for (auto& [pair, cnt] : future.get()) {
      merged[pair] += cnt;
    }
  }

  std::pair<std::uint32_t, std::uint32_t> best{0, 0};
  std::uint64_t best_count = 0;
  for (const auto& [pair, cnt] : merged) {
    if (cnt > best_count) {
      best = pair;
      best_count = cnt;
    }
  }

  if (best_count < options_.min_frequency) {
    return {std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max()};
  }
  return best;
}

void ByteLevelBPETrainer::MergePair(std::vector<std::vector<std::uint32_t>>& corpus,
                                    std::pair<std::uint32_t, std::uint32_t> pair,
                                    std::uint32_t new_id) const {
  for (auto& seq : corpus) {
    if (seq.size() < 2) {
      continue;
    }
    std::vector<std::uint32_t> merged;
    merged.reserve(seq.size());
    for (std::size_t i = 0; i < seq.size();) {
      if (i + 1 < seq.size() && seq[i] == pair.first && seq[i + 1] == pair.second) {
        merged.push_back(new_id);
        i += 2;
      } else {
        merged.push_back(seq[i]);
        ++i;
      }
    }
    seq.swap(merged);
  }
}

TrainResult ByteLevelBPETrainer::TrainFromLines(const std::vector<std::string>& lines) const {
  if (lines.empty()) {
    throw std::invalid_argument("training corpus is empty");
  }

  auto corpus = PretokenizeLines(lines);

  TrainResult result;
  result.vocab.reserve(options_.vocab_size);
  for (std::uint32_t i = 0; i < 256; ++i) {
    result.vocab.emplace_back(1, static_cast<char>(i));
  }

  std::uint32_t next_id = 256;
  while (result.vocab.size() < options_.vocab_size) {
    const auto best_pair = SelectBestPair(corpus);
    if (best_pair.first == std::numeric_limits<std::uint32_t>::max()) {
      break;
    }

    result.merges.emplace_back(result.vocab[best_pair.first], result.vocab[best_pair.second]);
    result.vocab.emplace_back(result.vocab[best_pair.first] + result.vocab[best_pair.second]);
    MergePair(corpus, best_pair, next_id);
    ++next_id;
  }

  return result;
}

TrainResult ByteLevelBPETrainer::TrainFromFiles(const std::vector<std::string>& files) const {
  std::vector<std::string> lines;
  lines.reserve(1 << 20);

  std::size_t accumulated = 0;
  for (const auto& file : files) {
    std::ifstream fs(file);
    if (!fs) {
      throw std::runtime_error("unable to read file: " + file);
    }

    std::string line;
    while (std::getline(fs, line)) {
      accumulated += line.size();
      if (accumulated > options_.max_memory_bytes) {
        break;
      }
      lines.push_back(line);
    }
    if (accumulated > options_.max_memory_bytes) {
      break;
    }
  }

  return TrainFromLines(lines);
}

}  // namespace tokenflux

