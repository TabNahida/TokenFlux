#pragma once

#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include "tokenflux/tokenizer.hpp"

namespace tokenflux {

class StreamingEncoder {
 public:
  explicit StreamingEncoder(const Tokenizer& tokenizer, std::size_t flush_chars = 4096)
      : tokenizer_(tokenizer), flush_chars_(flush_chars) {}

  void Push(std::string_view chunk);
  [[nodiscard]] std::vector<TokenId> Flush();

 private:
  const Tokenizer& tokenizer_;
  std::string buffer_;
  std::deque<TokenId> queue_;
  std::size_t flush_chars_;
};

}  // namespace tokenflux

