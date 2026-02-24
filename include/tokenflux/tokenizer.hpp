#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tokenflux {

struct TokenizerConfig {
    int unk_id = 0;
    std::string unk_token = "<unk>";
};

class Tokenizer {
public:
    Tokenizer();

    bool load_hf_tokenizer_json(const std::string& path);
    bool save_hf_tokenizer_json(const std::string& path) const;

    bool load_bpe_files(const std::string& vocab_json, const std::string& merges_txt);
    bool save_bpe_files(const std::string& vocab_json, const std::string& merges_txt) const;

    std::vector<int32_t> tokenize(std::string_view text) const;
    std::string detokenize(const std::vector<int32_t>& ids) const;

    int32_t token_to_id(std::string_view token) const;
    std::string id_to_token(int32_t id) const;

    void set_vocab(std::vector<std::string> vocab);
    void set_merges(std::vector<std::pair<std::string, std::string>> merges);
    const std::vector<std::string>& vocab() const { return vocab_; }

private:
    std::vector<std::string> byte_encode_word(std::string_view word) const;
    std::vector<std::string> apply_bpe(std::vector<std::string> pieces) const;
    void rebuild_index();

    TokenizerConfig cfg_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int32_t> merge_rank_;
};

} // namespace tokenflux
