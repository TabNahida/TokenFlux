#include "train_frontend.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
std::string to_lower_ascii(std::string s)
{
    for (char &c : s)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool parse_u64(const std::string &s, std::uint64_t &out)
{
    try
    {
        std::size_t pos = 0;
        std::uint64_t v = static_cast<std::uint64_t>(std::stoull(s, &pos, 10));
        if (pos != s.size())
        {
            return false;
        }
        out = v;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parse_size_value(const std::string &s, std::size_t &out)
{
    std::uint64_t v = 0;
    if (!parse_u64(s, v))
    {
        return false;
    }
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        return false;
    }
    out = static_cast<std::size_t>(v);
    return true;
}

bool parse_double_value(const std::string &s, double &out)
{
    try
    {
        std::size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size())
        {
            return false;
        }
        out = v;
        return std::isfinite(v);
    }
    catch (...)
    {
        return false;
    }
}

std::vector<std::string> split_csv(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == ',')
        {
            if (!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty())
    {
        out.push_back(cur);
    }
    return out;
}
} // namespace

std::string trainer_kind_to_string(TrainerKind kind)
{
    switch (kind)
    {
    case TrainerKind::byte_bpe:
        return "byte_bpe";
    case TrainerKind::bpe:
        return "bpe";
    case TrainerKind::wordpiece:
        return "wordpiece";
    case TrainerKind::unigram:
        return "unigram";
    }
    return "byte_bpe";
}

bool parse_trainer_kind(const std::string &text, TrainerKind &kind)
{
    std::string v = to_lower_ascii(text);
    if (v == "byte_bpe" || v == "byte-bpe" || v == "byte")
    {
        kind = TrainerKind::byte_bpe;
        return true;
    }
    if (v == "bpe")
    {
        kind = TrainerKind::bpe;
        return true;
    }
    if (v == "wordpiece" || v == "word_piece" || v == "wp")
    {
        kind = TrainerKind::wordpiece;
        return true;
    }
    if (v == "unigram" || v == "unigram_lm" || v == "unigramlm")
    {
        kind = TrainerKind::unigram;
        return true;
    }
    return false;
}

std::string detect_env_path_arg(int argc, char **argv, const std::string &default_path)
{
    std::string path = default_path;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--env" && i + 1 < argc)
        {
            path = argv[i + 1];
            ++i;
        }
    }
    return path;
}

void print_train_usage()
{
    std::cerr << "Tokenizer trainer (byte_bpe / bpe / wordpiece / unigram)\n"
              << "Usage:\n"
              << "  TokenFluxTrain [options]\n\n"
              << "Options:\n"
              << "  --env <path>                Path to .env (default: .env)\n"
              << "  --data <glob>               Data path glob (overrides DATA_PATH)\n"
              << "  --text-field <name>         JSON field name (default: text)\n"
              << "  --trainer <name>            byte_bpe | bpe | wordpiece | unigram\n"
              << "  --vocab-size <n>            Total vocab size incl. specials (default: 50000)\n"
              << "  --min-freq <n>              Min word frequency (default: 2)\n"
              << "  --min-pair-freq <n>         Min pair frequency for pair-based trainers (default: 2)\n"
              << "  --records-per-chunk <n>     Documents per chunk task (default: 5000)\n"
              << "  --chunk-docs <n>            In-task reduce cadence (default: 20000)\n"
              << "  --top-k <n>                 Keep top-k words per chunk (default: 200000)\n"
              << "  --chunk-dir <path>          Chunk output directory (default: artifacts/bpe/chunks)\n"
              << "  --resume / --no-resume      Resume from existing chunk files (default: on)\n"
              << "  --progress-interval <ms>    Progress update interval (default: 1000)\n"
              << "  --max-chars <n>             Truncate docs to N chars (default: 20000)\n"
              << "  --threads <n>               Worker threads (0=auto)\n"
              << "  --queue-capacity <n>        Task queue capacity (0=auto)\n"
              << "  --max-memory-mb <n>         Soft memory cap for counting/pairs (default: 0=unlimited)\n"
              << "  --pair-max-entries <n>      Max tracked pair keys (default: auto from --max-memory-mb)\n"
              << "  --max-token-length <n>      Max token length for unigram seed (default: 16)\n"
              << "  --unigram-iters <n>         EM iterations for unigram (default: 4)\n"
              << "  --unigram-seed-mult <n>     Seed vocab multiplier for unigram (default: 4)\n"
              << "  --unigram-prune-ratio <x>   Prune ratio for unigram in (0,1], default: 0.75\n"
              << "  --wordpiece-prefix <s>      WordPiece continuing prefix (default: ##)\n"
              << "  --output <path>             tokenizer.json output (default: tokenizer.json)\n"
              << "  --vocab <path>              vocab.json output (default: vocab.json)\n"
              << "  --merges <path>             merges.txt output (default: merges.txt)\n"
              << "  --no-vocab                  Do not write vocab.json\n"
              << "  --no-merges                 Do not write merges.txt\n"
              << "  --special-tokens <csv>      Comma-separated specials\n"
              << "  --help                      Show this help\n";
}

bool parse_train_args(int argc, char **argv, Config &cfg, std::string &err, bool &show_help)
{
    show_help = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto require_value = [&](const std::string &name) -> const char * {
            if (i + 1 >= argc)
            {
                err = "missing value for " + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            show_help = true;
            return false;
        }
        if (arg == "--env")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.env_path = v;
            continue;
        }
        if (arg == "--data")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.data_glob = v;
            continue;
        }
        if (arg == "--text-field")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.text_field = v;
            continue;
        }
        if (arg == "--trainer")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            TrainerKind kind = cfg.trainer;
            if (!parse_trainer_kind(v, kind))
            {
                err = "invalid --trainer: " + std::string(v);
                return false;
            }
            cfg.trainer = kind;
            continue;
        }
        if (arg == "--vocab-size")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.vocab_size))
            {
                err = "invalid --vocab-size";
                return false;
            }
            continue;
        }
        if (arg == "--min-freq")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.min_freq))
            {
                err = "invalid --min-freq";
                return false;
            }
            continue;
        }
        if (arg == "--min-pair-freq")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.min_pair_freq))
            {
                err = "invalid --min-pair-freq";
                return false;
            }
            continue;
        }
        if (arg == "--chunk-files")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.chunk_files))
            {
                err = "invalid --chunk-files";
                return false;
            }
            continue;
        }
        if (arg == "--chunk-docs")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.chunk_docs))
            {
                err = "invalid --chunk-docs";
                return false;
            }
            continue;
        }
        if (arg == "--records-per-chunk")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.records_per_chunk))
            {
                err = "invalid --records-per-chunk";
                return false;
            }
            continue;
        }
        if (arg == "--top-k")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.top_k))
            {
                err = "invalid --top-k";
                return false;
            }
            continue;
        }
        if (arg == "--chunk-dir")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.chunk_dir = v;
            continue;
        }
        if (arg == "--resume")
        {
            cfg.resume = true;
            continue;
        }
        if (arg == "--no-resume")
        {
            cfg.resume = false;
            continue;
        }
        if (arg == "--progress-interval")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.progress_interval_ms))
            {
                err = "invalid --progress-interval";
                return false;
            }
            continue;
        }
        if (arg == "--max-chars")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.max_chars_per_doc))
            {
                err = "invalid --max-chars";
                return false;
            }
            continue;
        }
        if (arg == "--threads")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.threads))
            {
                err = "invalid --threads";
                return false;
            }
            continue;
        }
        if (arg == "--queue-capacity")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.queue_capacity))
            {
                err = "invalid --queue-capacity";
                return false;
            }
            continue;
        }
        if (arg == "--max-memory-mb")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.max_memory_mb))
            {
                err = "invalid --max-memory-mb";
                return false;
            }
            continue;
        }
        if (arg == "--pair-max-entries")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.pair_max_entries))
            {
                err = "invalid --pair-max-entries";
                return false;
            }
            continue;
        }
        if (arg == "--max-token-length")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.max_token_length))
            {
                err = "invalid --max-token-length";
                return false;
            }
            continue;
        }
        if (arg == "--unigram-iters")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.unigram_em_iters))
            {
                err = "invalid --unigram-iters";
                return false;
            }
            continue;
        }
        if (arg == "--unigram-seed-mult")
        {
            const char *v = require_value(arg);
            if (!v || !parse_size_value(v, cfg.unigram_seed_multiplier))
            {
                err = "invalid --unigram-seed-mult";
                return false;
            }
            continue;
        }
        if (arg == "--unigram-prune-ratio")
        {
            const char *v = require_value(arg);
            if (!v || !parse_double_value(v, cfg.unigram_prune_ratio))
            {
                err = "invalid --unigram-prune-ratio";
                return false;
            }
            continue;
        }
        if (arg == "--wordpiece-prefix")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.wordpiece_continuing_prefix = v;
            continue;
        }
        if (arg == "--output")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.output_json = v;
            continue;
        }
        if (arg == "--vocab")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.output_vocab = v;
            continue;
        }
        if (arg == "--merges")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            cfg.output_merges = v;
            continue;
        }
        if (arg == "--no-vocab")
        {
            cfg.write_vocab = false;
            continue;
        }
        if (arg == "--no-merges")
        {
            cfg.write_merges = false;
            continue;
        }
        if (arg == "--special-tokens")
        {
            const char *v = require_value(arg);
            if (!v)
            {
                return false;
            }
            auto toks = split_csv(v);
            if (!toks.empty())
            {
                cfg.special_tokens = std::move(toks);
            }
            continue;
        }

        err = "unknown argument: " + arg;
        return false;
    }

    if (cfg.records_per_chunk == 0)
    {
        cfg.records_per_chunk = 1;
    }
    if (cfg.chunk_docs == 0)
    {
        cfg.chunk_docs = 1;
    }
    if (cfg.wordpiece_continuing_prefix.empty())
    {
        cfg.wordpiece_continuing_prefix = "##";
    }
    if (cfg.unigram_em_iters == 0)
    {
        cfg.unigram_em_iters = 1;
    }
    if (cfg.unigram_seed_multiplier == 0)
    {
        cfg.unigram_seed_multiplier = 1;
    }
    if (!(cfg.unigram_prune_ratio > 0.0 && cfg.unigram_prune_ratio <= 1.0))
    {
        err = "--unigram-prune-ratio must be in (0, 1]";
        return false;
    }

    return true;
}
