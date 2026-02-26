#pragma once

#include <string>

#include "trainers.h"

bool train_backend_byte_bpe(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                            std::string &err);
bool train_backend_bpe(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                       std::string &err);
bool train_backend_wordpiece(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                             std::string &err);
bool train_backend_unigram(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                           std::string &err);
