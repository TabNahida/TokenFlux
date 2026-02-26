#pragma once

#include <string>

#include "tokenflux_config.h"

std::string trainer_kind_to_string(TrainerKind kind);
bool parse_trainer_kind(const std::string &text, TrainerKind &kind);
std::string detect_env_path_arg(int argc, char **argv, const std::string &default_path = ".env");
void print_train_usage();
bool parse_train_args(int argc, char **argv, Config &cfg, std::string &err, bool &show_help);
