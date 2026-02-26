#include "tokenize_common.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize
{

bool starts_with(const std::string &s, const std::string &prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string normalize_path_for_compare(const std::string &path)
{
    std::string out = path;
    for (char &c : out)
    {
        if (c == '\\')
        {
            c = '/';
        }
#ifdef _WIN32
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
    }
    return out;
}

void print_usage()
{
    std::cerr << "TokenFluxTokenize: tokenize json/jsonl/json.gz/jsonl.gz/json.xz/jsonl.xz/parquet to train shards\n"
              << "Supports tokenizer.json model types: BPE / WordPiece / Unigram\n"
              << "Usage:\n"
              << "  TokenFluxTokenize [options]\n\n"
              << "Options:\n"
              << "  --env-file <path>            Path to .env (default: .env)\n"
              << "  --data-glob <glob>           Override DATA_PATH from .env\n"
              << "  --text-field <name>          JSON field name (default: text)\n"
              << "  --tokenizer <path>           tokenizer.json path (default: tokenizer.json)\n"
              << "  --out-dir <path>             Output directory (default: data/tokens)\n"
              << "  --max-tokens-per-shard <n>   Tokens per train shard (default: 50000000)\n"
              << "  --encode-batch-size <n>      Docs per in-file encode batch (default: 256)\n"
              << "  --min-chars <n>              Min chars per doc (default: 1)\n"
              << "  --max-chars <n>              Max chars per doc (default: 20000)\n"
              << "  --max-docs <n>               CLI compatibility only (must be 0)\n"
              << "  --eos-token <tok>            EOS token (default: </s>)\n"
              << "  --bos-token <tok>            BOS token (default: none)\n"
              << "  --threads <n>                Worker threads (0=auto)\n"
              << "  --cache-max-entries <n>      Max in-memory token-piece cache entries per worker (default: 50000, 0=disable)\n"
              << "  --max-memory-mb <n>          Soft memory cap for per-file processing (default: 0=unlimited)\n"
              << "  --prescan / --no-prescan     Pre-scan docs for stable ETA/docs total (default: on)\n"
              << "  --resume / --no-resume       Reuse completed-file list for resume (default: on)\n"
              << "  --progress-every <n>         CLI compatibility only (unused)\n"
              << "  --help                       Show this help\n";
}

bool parse_u64_arg(const std::string &s, std::uint64_t &out)
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

bool parse_size_arg(const std::string &s, std::size_t &out)
{
    std::uint64_t v = 0;
    if (!parse_u64_arg(s, v))
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

bool parse_args(int argc, char **argv, Args &args)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto need_value = [&](const std::string &name) -> const char * {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return false;
        }
        if (arg == "--")
        {
            continue;
        }
        if (arg == "--env-file")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            args.env_file = v;
            continue;
        }
        if (arg == "--data-glob")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            args.data_glob = v;
            continue;
        }
        if (arg == "--text-field")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            args.text_field = v;
            continue;
        }
        if (arg == "--tokenizer")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            args.tokenizer_path = v;
            continue;
        }
        if (arg == "--out-dir")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            args.out_dir = v;
            continue;
        }
        if (arg == "--max-tokens-per-shard")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::uint64_t x = 0;
            if (!parse_u64_arg(v, x) || x == 0)
            {
                std::cerr << "Invalid --max-tokens-per-shard: " << v << "\n";
                return false;
            }
            args.max_tokens_per_shard = x;
            continue;
        }
        if (arg == "--encode-batch-size")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --encode-batch-size: " << v << "\n";
                return false;
            }
            args.encode_batch_size = x;
            continue;
        }
        if (arg == "--min-chars")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --min-chars: " << v << "\n";
                return false;
            }
            args.min_chars = x;
            continue;
        }
        if (arg == "--max-chars")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --max-chars: " << v << "\n";
                return false;
            }
            args.max_chars = x;
            continue;
        }
        if (arg == "--max-docs")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::uint64_t x = 0;
            if (!parse_u64_arg(v, x))
            {
                std::cerr << "Invalid --max-docs: " << v << "\n";
                return false;
            }
            args.max_docs = x;
            continue;
        }
        if (arg == "--eos-token")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            args.eos_token = v;
            continue;
        }
        if (arg == "--bos-token")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            args.bos_token = v;
            continue;
        }
        if (arg == "--threads")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --threads: " << v << "\n";
                return false;
            }
            args.threads = x;
            continue;
        }
        if (arg == "--cache-max-entries")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --cache-max-entries: " << v << "\n";
                return false;
            }
            args.cache_max_entries = x;
            continue;
        }
        if (arg == "--max-memory-mb")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --max-memory-mb: " << v << "\n";
                return false;
            }
            args.max_memory_mb = x;
            continue;
        }
        if (arg == "--prescan")
        {
            args.prescan_records = true;
            continue;
        }
        if (arg == "--no-prescan")
        {
            args.prescan_records = false;
            continue;
        }
        if (arg == "--resume")
        {
            args.resume = true;
            continue;
        }
        if (arg == "--no-resume")
        {
            args.resume = false;
            continue;
        }
        if (arg == "--progress-every")
        {
            const char *v = need_value(arg);
            if (!v)
            {
                return false;
            }
            std::uint64_t x = 0;
            if (!parse_u64_arg(v, x))
            {
                std::cerr << "Invalid --progress-every: " << v << "\n";
                return false;
            }
            args.progress_every = x;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage();
        return false;
    }
    return true;
}

std::vector<std::string> expand_data_files(const std::string &data_glob)
{
    return expand_data_glob(data_glob);
}

void append_utf8(std::uint32_t cp, std::string &out)
{
    if (cp <= 0x7F)
    {
        out.push_back(static_cast<char>(cp));
    }
    else if (cp <= 0x7FF)
    {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else if (cp <= 0xFFFF)
    {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

static bool is_unicode_space(std::uint32_t cp)
{
    if (cp <= 0x20)
    {
        return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D || cp == 0x20;
    }
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || cp == 0x180E || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

std::vector<std::string> split_whitespace_words(const std::string &text)
{
    std::vector<std::string> out;
    std::string cur;
    std::size_t i = 0;
    while (i < text.size())
    {
        std::uint32_t cp = 0;
        std::size_t prev = i;
        if (!next_codepoint(text, i, cp))
        {
            break;
        }
        if (i <= prev)
        {
            break;
        }
        std::string cp_utf8;
        append_utf8(cp, cp_utf8);
        if (is_unicode_space(cp))
        {
            if (!cur.empty())
            {
                out.push_back(std::move(cur));
                cur.clear();
            }
            continue;
        }
        cur += cp_utf8;
    }
    if (!cur.empty())
    {
        out.push_back(std::move(cur));
    }
    return out;
}

std::vector<std::string> split_codepoints_utf8(const std::string &text)
{
    std::vector<std::string> cps;
    std::size_t i = 0;
    while (i < text.size())
    {
        std::uint32_t cp = 0;
        std::size_t prev = i;
        if (!next_codepoint(text, i, cp))
        {
            break;
        }
        if (i <= prev)
        {
            break;
        }
        std::string cp_utf8;
        append_utf8(cp, cp_utf8);
        cps.push_back(std::move(cp_utf8));
    }
    return cps;
}

std::string json_escape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            }
            else
            {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    return out;
}

std::string read_file_all(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace tokenflux::tokenize
