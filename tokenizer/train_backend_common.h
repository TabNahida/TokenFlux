#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenflux_bpe.h"
#include "tokenflux_config.h"
#include "train_io.h"

using SymbolMap = std::unordered_map<std::string, int>;

bool starts_with(const std::string &s, const std::string &prefix);
void append_utf8(uint32_t cp, std::string &out);
std::vector<std::string> split_whitespace_words(const std::string &text);
std::vector<std::string> split_codepoints(const std::string &text);
std::string json_escape(const std::string &s);
std::vector<std::string> make_special_tokens(const Config &cfg);
std::size_t calc_pair_target_vocab(const Config &cfg, std::size_t base_symbols, std::size_t special_count);
void maybe_reduce_local(LocalCountMap &local_counts, const Config &cfg, std::size_t &reduce_counter,
                        std::size_t local_entry_cap);
int ensure_symbol(const std::string &token, SymbolMap &token_to_id, std::vector<std::string> &id_to_token);
std::vector<Word> build_words_from_tokens(const GlobalCountMap &global_counts,
                                          const std::unordered_map<uint32_t, int> &cp_to_id, std::size_t min_freq);
void append_symbols_to_vocab(const std::vector<std::string> &specials, const std::vector<std::string> &symbols,
                             std::vector<std::string> &id_to_token);
uint64_t pair_key(int a, int b);
