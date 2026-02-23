#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokenflux {

using TokenId = std::uint32_t;

struct EncodeOptions {
  bool add_bos = false;
  bool add_eos = false;
};

struct DecodeOptions {
  bool skip_special_tokens = true;
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  [[nodiscard]] virtual std::vector<TokenId> Encode(std::string_view text,
                                                     EncodeOptions opts = {}) const = 0;
  [[nodiscard]] virtual std::string Decode(std::span<const TokenId> ids,
                                           DecodeOptions opts = {}) const = 0;

  [[nodiscard]] virtual std::size_t VocabSize() const = 0;
  [[nodiscard]] virtual std::string_view TokenById(TokenId id) const = 0;
  [[nodiscard]] virtual TokenId IdByToken(std::string_view token) const = 0;
};

using Vocab = std::vector<std::string>;
using MergeRules = std::vector<std::pair<std::string, std::string>>;

class ByteLevelBPETokenizer final : public Tokenizer {
 public:
  ByteLevelBPETokenizer(Vocab vocab, MergeRules merges, TokenId unk_id,
                        TokenId bos_id = static_cast<TokenId>(-1),
                        TokenId eos_id = static_cast<TokenId>(-1));

  static ByteLevelBPETokenizer FromFiles(const std::string& vocab_path,
                                         const std::string& merges_path,
                                         TokenId unk_id,
                                         TokenId bos_id = static_cast<TokenId>(-1),
                                         TokenId eos_id = static_cast<TokenId>(-1));

  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text,
                                            EncodeOptions opts = {}) const override;
  [[nodiscard]] std::string Decode(std::span<const TokenId> ids,
                                   DecodeOptions opts = {}) const override;

  [[nodiscard]] std::size_t VocabSize() const override { return vocab_.size(); }
  [[nodiscard]] std::string_view TokenById(TokenId id) const override;
  [[nodiscard]] TokenId IdByToken(std::string_view token) const override;

  [[nodiscard]] const Vocab& GetVocab() const { return vocab_; }
  [[nodiscard]] const MergeRules& GetMerges() const { return merges_; }

 private:
  std::vector<std::string> SplitToByteTokens(std::string_view text) const;
  std::vector<std::string> ApplyMerges(std::vector<std::string> tokens) const;

  Vocab vocab_;
  MergeRules merges_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::string, std::size_t> merge_rank_;
  TokenId unk_id_;
  TokenId bos_id_;
  TokenId eos_id_;
};

}  // namespace tokenflux

