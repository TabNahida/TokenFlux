#include "tokenflux/stream.hpp"

namespace tokenflux {

void StreamingEncoder::Push(std::string_view chunk) {
  buffer_.append(chunk);
  if (buffer_.size() < flush_chars_) {
    return;
  }

  auto ids = tokenizer_.Encode(buffer_);
  for (auto id : ids) {
    queue_.push_back(id);
  }
  buffer_.clear();
}

std::vector<TokenId> StreamingEncoder::Flush() {
  if (!buffer_.empty()) {
    auto ids = tokenizer_.Encode(buffer_);
    for (auto id : ids) {
      queue_.push_back(id);
    }
    buffer_.clear();
  }

  std::vector<TokenId> out;
  out.reserve(queue_.size());
  while (!queue_.empty()) {
    out.push_back(queue_.front());
    queue_.pop_front();
  }
  return out;
}

}  // namespace tokenflux

