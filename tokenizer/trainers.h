#pragma once

#include <string>
#include <vector>

#include "tokenflux_config.h"
#include "train_io.h"

struct TrainArtifacts
{
    std::vector<std::string> id_to_token;
    std::vector<std::string> merges;
    std::vector<double> token_scores;
    bool has_merges = false;
};

ProcessTextFn build_process_text_callback(const Config &cfg);
bool train_from_global_counts(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                              std::string &err);
bool write_trained_tokenizer(const Config &cfg, const TrainArtifacts &artifacts, std::string &err);

