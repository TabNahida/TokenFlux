#pragma once

#include <cstddef>
#include <string>
#include <vector>

enum class TrainerKind
{
    byte_bpe = 0,
    bpe,
    wordpiece,
    unigram
};

struct Config
{
    std::string env_path = ".env";
    std::string data_glob;
    std::string data_list;
    std::string text_field = "text";
    std::string output_json = "tokenizer.json";
    std::string output_vocab = "vocab.json";
    std::string output_merges = "merges.txt";
    std::string unk_token = "<unk>";
    std::vector<std::string> special_tokens = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"};
    TrainerKind trainer = TrainerKind::byte_bpe;

    std::size_t vocab_size = 50000;
    std::size_t min_freq = 2;
    std::size_t min_pair_freq = 2;
    std::size_t chunk_files = 1;
    std::size_t chunk_docs = 20000; // in-chunk reduce cadence
    std::size_t top_k = 200000;
    std::size_t max_chars_per_doc = 20000;
    std::size_t threads = 0; // 0 -> auto
    std::size_t progress_interval_ms = 1000;
    std::size_t max_memory_mb = 0;        // 0 -> unlimited
    std::size_t pair_max_entries = 0;     // 0 -> unlimited (or derived from max_memory_mb)
    std::size_t records_per_chunk = 5000; // granularity for chunk write + progress
    std::size_t queue_capacity = 0;       // 0 -> derived from threads
    std::size_t max_token_length = 16;    // used by unigram trainer
    std::size_t unigram_em_iters = 4;
    std::size_t unigram_seed_multiplier = 4;
    double unigram_prune_ratio = 0.75;
    std::string wordpiece_continuing_prefix = "##";
    bool prescan_records = false; // off by default for single-pass streaming

    std::string chunk_dir = "artifacts/bpe/chunks";
    bool resume = true;
    bool write_vocab = true;
    bool write_merges = true;
};
