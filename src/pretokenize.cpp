#include "tokenflux/pretokenize.hpp"

#include <algorithm>
#include <fstream>
#include <future>
#include <stdexcept>
#include <thread>

namespace tokenflux {

namespace {
std::size_t EffectiveThreads(std::size_t configured) {
  if (configured > 0) {
    return configured;
  }
  const auto hw = std::thread::hardware_concurrency();
  return hw == 0 ? 4 : hw;
}
}  // namespace

PretokenizeEngine::PretokenizeEngine(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

void PretokenizeEngine::PretokenizeFiles(const std::vector<std::string>& input_files,
                                         const std::string& output_bin,
                                         const std::string& output_index,
                                         PretokenizeOptions options) const {
  std::vector<std::string> lines;
  for (const auto& file : input_files) {
    std::ifstream in(file);
    if (!in) {
      throw std::runtime_error("failed to open input file: " + file);
    }
    std::string line;
    while (std::getline(in, line)) {
      lines.push_back(std::move(line));
    }
  }

  std::vector<std::vector<TokenId>> encoded(lines.size());
  for (std::size_t idx = 0; idx < lines.size(); ++idx) {
    encoded[idx] = tokenizer_.Encode(lines[idx]);
  }

  std::ofstream out(output_bin, std::ios::binary);
  std::ofstream index(output_index, std::ios::binary);
  if (!out || !index) {
    throw std::runtime_error("failed to create output files");
  }

  std::uint64_t cursor = 0;
  for (const auto& seq : encoded) {
    PretokenizeRecord rec{cursor, static_cast<std::uint32_t>(seq.size())};
    out.write(reinterpret_cast<const char*>(seq.data()),
              static_cast<std::streamsize>(seq.size() * sizeof(TokenId)));
    index.write(reinterpret_cast<const char*>(&rec), static_cast<std::streamsize>(sizeof(PretokenizeRecord)));
    cursor += seq.size() * sizeof(TokenId);
  }
}

}  // namespace tokenflux

