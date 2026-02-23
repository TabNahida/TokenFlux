#include "tokenflux/tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tokenflux {
namespace {
constexpr TokenId kInvalidId = std::numeric_limits<TokenId>::max();
}

ByteLevelBPETokenizer::ByteLevelBPETokenizer(Vocab vocab, MergeRules merges, TokenId unk_id,
                                             TokenId bos_id, TokenId eos_id)
    : vocab_(std::move(vocab)), merges_(std::move(merges)), unk_id_(unk_id), bos_id_(bos_id), eos_id_(eos_id) {
  for (TokenId i = 0; i < vocab_.size(); ++i) {
    token_to_id_.emplace(vocab_[i], i);
  }
  for (std::size_t i = 0; i < merges_.size(); ++i) {
    merge_rank_.emplace(merges_[i].first + "\x1f" + merges_[i].second, i);
  }
}

ByteLevelBPETokenizer ByteLevelBPETokenizer::FromFiles(const std::string& vocab_path,
                                                       const std::string& merges_path,
                                                       TokenId unk_id,
                                                       TokenId bos_id,
                                                       TokenId eos_id) {
  std::ifstream vocab_fs(vocab_path);
  std::ifstream merges_fs(merges_path);
  if (!vocab_fs || !merges_fs) {
    throw std::runtime_error("failed to open tokenizer files");
  }

  Vocab vocab;
  std::string line;
  while (std::getline(vocab_fs, line)) {
    if (!line.empty()) {
      vocab.push_back(line);
    }
  }

  MergeRules merges;
  while (std::getline(merges_fs, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    std::string left;
    std::string right;
    if (iss >> left >> right) {
      merges.emplace_back(std::move(left), std::move(right));
    }
  }

  return ByteLevelBPETokenizer(std::move(vocab), std::move(merges), unk_id, bos_id, eos_id);
}

std::vector<std::string> ByteLevelBPETokenizer::SplitToByteTokens(std::string_view text) const {
  std::vector<std::string> tokens;
  tokens.reserve(text.size());
  for (unsigned char c : text) {
    tokens.emplace_back(1, static_cast<char>(c));
  }
  return tokens;
}

std::vector<std::string> ByteLevelBPETokenizer::ApplyMerges(std::vector<std::string> tokens) const {
  if (tokens.size() < 2 || merge_rank_.empty()) {
    return tokens;
  }

  while (tokens.size() > 1) {
    std::size_t best_pos = tokens.size();
    std::size_t best_rank = merge_rank_.size();

    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      auto it = merge_rank_.find(tokens[i] + "\x1f" + tokens[i + 1]);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pos = i;
      }
    }

    if (best_pos == tokens.size()) {
      break;
    }

    tokens[best_pos] += tokens[best_pos + 1];
    tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_pos + 1));
  }

  return tokens;
}

std::vector<TokenId> ByteLevelBPETokenizer::Encode(std::string_view text, EncodeOptions opts) const {
  std::vector<TokenId> ids;
  if (opts.add_bos && bos_id_ != kInvalidId) {
    ids.push_back(bos_id_);
  }

  auto tokens = ApplyMerges(SplitToByteTokens(text));
  ids.reserve(ids.size() + tokens.size() + 1);
  for (const auto& token : tokens) {
    auto it = token_to_id_.find(token);
    ids.push_back(it == token_to_id_.end() ? unk_id_ : it->second);
  }

  if (opts.add_eos && eos_id_ != kInvalidId) {
    ids.push_back(eos_id_);
  }
  return ids;
}

std::string ByteLevelBPETokenizer::Decode(std::span<const TokenId> ids, DecodeOptions opts) const {
  std::string out;
  for (auto id : ids) {
    if (id >= vocab_.size()) {
      continue;
    }
    if (opts.skip_special_tokens && (id == bos_id_ || id == eos_id_)) {
      continue;
    }
    out += vocab_[id];
  }
  return out;
}

std::string_view ByteLevelBPETokenizer::TokenById(TokenId id) const {
  if (id >= vocab_.size()) {
    throw std::out_of_range("token id out of range");
  }
  return vocab_[id];
}

TokenId ByteLevelBPETokenizer::IdByToken(std::string_view token) const {
  auto it = token_to_id_.find(std::string(token));
  return it == token_to_id_.end() ? unk_id_ : it->second;
}

}  // namespace tokenflux

