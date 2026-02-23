#include "tokenflux/formats.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace tokenflux {

void SaveAsGPT2(const TrainResult& result,
                const std::string& vocab_path,
                const std::string& merges_path) {
  std::ofstream vocab(vocab_path);
  std::ofstream merges(merges_path);
  if (!vocab || !merges) {
    throw std::runtime_error("failed to create tokenizer output files");
  }

  for (const auto& token : result.vocab) {
    vocab << token << '\n';
  }
  merges << "#version: 0.2\n";
  for (const auto& pair : result.merges) {
    merges << pair.first << ' ' << pair.second << '\n';
  }
}

void SaveAsHFTokenizerJson(const TrainResult& result,
                           const std::string& tokenizer_json_path,
                           std::string model_type) {
  nlohmann::json j;
  j["version"] = "1.0";
  j["truncation"] = nullptr;
  j["padding"] = nullptr;
  j["added_tokens"] = nlohmann::json::array();
  j["normalizer"] = nullptr;
  j["pre_tokenizer"] = { {"type", "ByteLevel"}, {"add_prefix_space", false} };
  j["post_processor"] = nullptr;
  j["decoder"] = { {"type", "ByteLevel"} };

  nlohmann::json vocab_json;
  for (std::size_t i = 0; i < result.vocab.size(); ++i) {
    vocab_json[result.vocab[i]] = i;
  }

  nlohmann::json merges_json = nlohmann::json::array();
  for (const auto& pair : result.merges) {
    merges_json.push_back({pair.first, pair.second});
  }

  j["model"] = {
      {"type", std::move(model_type)},
      {"dropout", nullptr},
      {"unk_token", "<unk>"},
      {"continuing_subword_prefix", ""},
      {"end_of_word_suffix", ""},
      {"fuse_unk", false},
      {"vocab", std::move(vocab_json)},
      {"merges", std::move(merges_json)},
  };

  std::ofstream out(tokenizer_json_path);
  if (!out) {
    throw std::runtime_error("failed to create tokenizer json");
  }
  out << j.dump(2);
}

}  // namespace tokenflux

