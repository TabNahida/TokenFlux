#pragma once

#include <string>

#include "tokenflux/tokenizer.hpp"
#include "tokenflux/trainer.hpp"

namespace tokenflux {

void SaveAsGPT2(const TrainResult& result,
                const std::string& vocab_path,
                const std::string& merges_path);

void SaveAsHFTokenizerJson(const TrainResult& result,
                           const std::string& tokenizer_json_path,
                           std::string model_type = "BPE");

}  // namespace tokenflux

