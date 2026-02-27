#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokenflux
{
namespace tokenize
{

struct Args
{
    std::string env_file = ".env";
    std::string data_glob;
    std::string data_list;
    std::vector<std::string> input_entries;
    std::string text_field = "text";
    std::string tokenizer_path = "tokenizer.json";
    std::string out_dir = "data/tokens";
    std::uint64_t max_tokens_per_shard = 50'000'000;
    std::size_t encode_batch_size = 256;
    std::size_t min_chars = 1;
    std::size_t max_chars = 20'000;
    std::uint64_t max_docs = 0;
    std::string eos_token = "<|endoftext|>";
    std::string bos_token;
    std::uint64_t progress_every = 10'000;
    std::size_t threads = 0;
    std::size_t cache_max_entries = 50'000;
    std::size_t max_memory_mb = 0;
    bool prescan_records = false;
    bool resume = true;
};

enum class ModelType
{
    bpe = 0,
    wordpiece,
    unigram
};

enum class PretokenizerType
{
    byte_level = 0,
    whitespace
};

struct UnigramEntry
{
    std::string token;
    double score = 0.0;
};

struct TokenizerData
{
    std::unordered_map<std::string, std::uint32_t> vocab;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, std::uint32_t> added_tokens;
    std::string unk_token;
    std::string continuing_subword_prefix = "##";
    ModelType model_type = ModelType::bpe;
    PretokenizerType pretokenizer_type = PretokenizerType::byte_level;
    std::vector<UnigramEntry> unigram_vocab;
};

struct MergeRule
{
    std::uint32_t rank = 0;
    std::uint32_t merged_symbol = 0;
};

struct FileTask
{
    std::size_t index = 0;
    std::string source;
    std::string path;
    std::string normalized_path;
    std::uint64_t file_size = 0;
    std::int64_t file_mtime = 0;
};

struct PartResult
{
    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::uint64_t num_tokens = 0;
    bool reused = false;
};

struct ShardInfo
{
    std::string file;
    std::uint64_t num_tokens = 0;
};

struct CompletionRecord
{
    std::uint64_t source_size = 0;
    std::int64_t source_mtime = 0;
    PartResult result;
};

struct PartSignature
{
    std::string tokenizer_fingerprint;
    std::string text_field;
    std::size_t min_chars = 1;
    std::size_t max_chars = 20'000;
    std::int64_t bos_id = -1;
    std::int64_t eos_id = -1;
    std::uint32_t dtype_bytes = 2;
};

bool starts_with(const std::string &s, const std::string &prefix);
bool ends_with(const std::string &s, const std::string &suffix);
std::string normalize_path_for_compare(const std::string &path);

void print_usage();
bool parse_u64_arg(const std::string &s, std::uint64_t &out);
bool parse_size_arg(const std::string &s, std::size_t &out);
bool parse_args(int argc, char **argv, Args &args);
std::vector<std::string> expand_data_files(const std::string &data_glob);

void append_utf8(std::uint32_t cp, std::string &out);
std::vector<std::string> split_whitespace_words(const std::string &text);
std::vector<std::string> split_codepoints_utf8(const std::string &text);

std::string json_escape(const std::string &s);
std::string read_file_all(const std::string &path);

} // namespace tokenize
} // namespace tokenflux
