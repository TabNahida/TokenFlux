#include "tokenflux_lib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <zlib.h>
#include <lzma.h>

static std::string trim(const std::string &s)
{
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
    {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
    {
        --end;
    }
    return s.substr(start, end - start);
}

static std::vector<std::string> split_csv(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == ',')
        {
            out.push_back(trim(cur));
            cur.clear();
        }
        else
        {
            cur.push_back(c);
        }
    }
    if (!cur.empty() || !out.empty())
    {
        out.push_back(trim(cur));
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const std::string &v) { return v.empty(); }), out.end());
    return out;
}

static std::size_t parse_size(const std::string &s, std::size_t def_val)
{
    try
    {
        return static_cast<std::size_t>(std::stoull(s));
    }
    catch (...)
    {
        return def_val;
    }
}

static double parse_double(const std::string &s, double def_val)
{
    try
    {
        return std::stod(s);
    }
    catch (...)
    {
        return def_val;
    }
}

static bool parse_bool(const std::string &s, bool def_val)
{
    std::string v;
    v.reserve(s.size());
    for (char c : s)
    {
        v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (v == "1" || v == "true" || v == "yes" || v == "y" || v == "on")
    {
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "n" || v == "off")
    {
        return false;
    }
    return def_val;
}

static TrainerKind parse_trainer_kind(const std::string &s, TrainerKind def_val)
{
    std::string v;
    v.reserve(s.size());
    for (char c : s)
    {
        v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (v == "byte_bpe" || v == "byte-bpe" || v == "byte")
    {
        return TrainerKind::byte_bpe;
    }
    if (v == "bpe")
    {
        return TrainerKind::bpe;
    }
    if (v == "wordpiece" || v == "word_piece" || v == "wp")
    {
        return TrainerKind::wordpiece;
    }
    if (v == "unigram" || v == "unigram_lm" || v == "unigramlm")
    {
        return TrainerKind::unigram;
    }
    return def_val;
}

std::unordered_map<std::string, std::string> read_env_file(const std::string &path)
{
    std::unordered_map<std::string, std::string> env;
    std::ifstream in(path);
    if (!in)
    {
        return env;
    }
    bool first_line = true;
    std::string line;
    while (std::getline(in, line))
    {
        if (first_line)
        {
            first_line = false;
            if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB && static_cast<unsigned char>(line[2]) == 0xBF)
            {
                line.erase(0, 3);
            }
        }
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        auto trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
        {
            continue;
        }
        auto eq = trimmed.find('=');
        if (eq == std::string::npos)
        {
            continue;
        }
        std::string key = trim(trimmed.substr(0, eq));
        std::string val = trim(trimmed.substr(eq + 1));
        if (val.size() >= 2 &&
            ((val.front() == '"' && val.back() == '"') || (val.front() == '\'' && val.back() == '\'')))
        {
            val = val.substr(1, val.size() - 2);
        }
        env[key] = val;
    }
    return env;
}

void apply_env_overrides(Config &cfg, const std::unordered_map<std::string, std::string> &env)
{
    auto get = [&](const std::string &key) -> const std::string * {
        auto it = env.find(key);
        if (it == env.end())
        {
            return nullptr;
        }
        return &it->second;
    };
    if (auto v = get("DATA_PATH"))
        cfg.data_glob = *v;
    if (auto v = get("DATA_GLOB"))
        cfg.data_glob = *v;
    if (auto v = get("TEXT_FIELD"))
        cfg.text_field = *v;
    if (auto v = get("VOCAB_SIZE"))
        cfg.vocab_size = parse_size(*v, cfg.vocab_size);
    if (auto v = get("MIN_FREQ"))
        cfg.min_freq = parse_size(*v, cfg.min_freq);
    if (auto v = get("MIN_PAIR_FREQ"))
        cfg.min_pair_freq = parse_size(*v, cfg.min_pair_freq);
    if (auto v = get("TRAINER"))
        cfg.trainer = parse_trainer_kind(*v, cfg.trainer);
    if (auto v = get("TRAINER_KIND"))
        cfg.trainer = parse_trainer_kind(*v, cfg.trainer);
    if (auto v = get("CHUNK_FILES"))
        cfg.chunk_files = parse_size(*v, cfg.chunk_files);
    if (auto v = get("CHUNK_DOCS"))
        cfg.chunk_docs = parse_size(*v, cfg.chunk_docs);
    if (auto v = get("RECORDS_PER_CHUNK"))
        cfg.records_per_chunk = parse_size(*v, cfg.records_per_chunk);
    if (auto v = get("QUEUE_CAPACITY"))
        cfg.queue_capacity = parse_size(*v, cfg.queue_capacity);
    if (auto v = get("TOP_K"))
        cfg.top_k = parse_size(*v, cfg.top_k);
    if (auto v = get("MAX_CHARS_PER_DOC"))
        cfg.max_chars_per_doc = parse_size(*v, cfg.max_chars_per_doc);
    if (auto v = get("MAX_TOKEN_LENGTH"))
        cfg.max_token_length = parse_size(*v, cfg.max_token_length);
    if (auto v = get("THREADS"))
        cfg.threads = parse_size(*v, cfg.threads);
    if (auto v = get("MAX_MEMORY_MB"))
        cfg.max_memory_mb = parse_size(*v, cfg.max_memory_mb);
    if (auto v = get("PAIR_MAX_ENTRIES"))
        cfg.pair_max_entries = parse_size(*v, cfg.pair_max_entries);
    if (auto v = get("OUTPUT_JSON"))
        cfg.output_json = *v;
    if (auto v = get("OUTPUT_VOCAB"))
        cfg.output_vocab = *v;
    if (auto v = get("OUTPUT_MERGES"))
        cfg.output_merges = *v;
    if (auto v = get("UNK_TOKEN"))
        cfg.unk_token = *v;
    if (auto v = get("CHUNK_DIR"))
        cfg.chunk_dir = *v;
    if (auto v = get("RESUME"))
        cfg.resume = parse_bool(*v, cfg.resume);
    if (auto v = get("PROGRESS_INTERVAL"))
        cfg.progress_interval_ms = parse_size(*v, cfg.progress_interval_ms);
    if (auto v = get("PROGRESS_INTERVAL_MS"))
        cfg.progress_interval_ms = parse_size(*v, cfg.progress_interval_ms);
    if (auto v = get("WORDPIECE_CONTINUING_PREFIX"))
        cfg.wordpiece_continuing_prefix = *v;
    if (auto v = get("UNIGRAM_EM_ITERS"))
        cfg.unigram_em_iters = parse_size(*v, cfg.unigram_em_iters);
    if (auto v = get("UNIGRAM_SEED_MULTIPLIER"))
        cfg.unigram_seed_multiplier = parse_size(*v, cfg.unigram_seed_multiplier);
    if (auto v = get("UNIGRAM_PRUNE_RATIO"))
        cfg.unigram_prune_ratio = parse_double(*v, cfg.unigram_prune_ratio);
    if (auto v = get("SPECIAL_TOKENS"))
    {
        auto tokens = split_csv(*v);
        if (!tokens.empty())
        {
            cfg.special_tokens = std::move(tokens);
        }
    }
}

static std::string normalize_path(const std::string &s)
{
    std::string out = s;
    for (auto &c : out)
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

static bool wildcard_match(const std::string &pattern, const std::string &str)
{
    std::size_t p = 0;
    std::size_t s = 0;
    std::size_t star = std::string::npos;
    std::size_t match = 0;
    while (s < str.size())
    {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == str[s]))
        {
            ++p;
            ++s;
        }
        else if (p < pattern.size() && pattern[p] == '*')
        {
            star = p++;
            match = s;
        }
        else if (star != std::string::npos)
        {
            p = star + 1;
            s = ++match;
        }
        else
        {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*')
    {
        ++p;
    }
    return p == pattern.size();
}

std::vector<std::string> glob_files(const std::string &pattern)
{
    std::vector<std::string> out;
    std::string norm_pattern = normalize_path(pattern);
    auto first_wild = norm_pattern.find_first_of("*?");
    if (first_wild == std::string::npos)
    {
        if (std::filesystem::exists(pattern))
        {
            out.push_back(pattern);
        }
        return out;
    }
    auto last_sep = norm_pattern.substr(0, first_wild).find_last_of('/');
    std::filesystem::path base_dir = ".";
    if (last_sep != std::string::npos)
    {
        base_dir = pattern.substr(0, last_sep);
    }
    std::error_code ec;
    if (!std::filesystem::exists(base_dir, ec))
    {
        return out;
    }
    for (std::filesystem::recursive_directory_iterator it(base_dir, ec), end; it != end; it.increment(ec))
    {
        if (ec)
        {
            continue;
        }
        if (!it->is_regular_file())
        {
            continue;
        }
        std::string cand = normalize_path(it->path().string());
        if (wildcard_match(norm_pattern, cand))
        {
            out.push_back(it->path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static bool ends_with(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> expand_data_glob(const std::string &pattern)
{
    auto files = glob_files(pattern);
    if (!files.empty())
    {
        return files;
    }

    static const std::array<std::string, 8> suffixes = {
        ".json.gz", ".jsonl.gz", ".json.xz", ".jsonl.xz", ".json", ".jsonl", ".ndjson", ".parquet"};
    std::string norm = normalize_path(pattern);
    std::vector<std::string> candidates;
    candidates.reserve(suffixes.size() + 2);

    bool replaced = false;
    for (const auto &suffix : suffixes)
    {
        if (ends_with(norm, suffix))
        {
            std::string base = pattern.substr(0, pattern.size() - suffix.size());
            for (const auto &alt : suffixes)
            {
                if (alt != suffix)
                {
                    candidates.push_back(base + alt);
                }
            }
            replaced = true;
            break;
        }
    }
    if (!replaced)
    {
        for (const auto &suffix : suffixes)
        {
            candidates.push_back(pattern + suffix);
        }
    }

    std::unordered_set<std::string> seen;
    for (const auto &cand : candidates)
    {
        auto partial = glob_files(cand);
        for (const auto &path : partial)
        {
            std::string norm_path = normalize_path(path);
            if (seen.insert(norm_path).second)
            {
                files.push_back(path);
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

InputFormat detect_input_format(const std::string &path)
{
    std::string norm = normalize_path(path);
    if (ends_with(norm, ".jsonl.gz") || ends_with(norm, ".ndjson.gz"))
    {
        return InputFormat::jsonl_gz;
    }
    if (ends_with(norm, ".json.gz"))
    {
        return InputFormat::json_gz;
    }
    if (ends_with(norm, ".jsonl.xz") || ends_with(norm, ".ndjson.xz"))
    {
        return InputFormat::jsonl_xz;
    }
    if (ends_with(norm, ".json.xz"))
    {
        return InputFormat::json_xz;
    }
    if (ends_with(norm, ".jsonl") || ends_with(norm, ".ndjson"))
    {
        return InputFormat::jsonl;
    }
    if (ends_with(norm, ".json"))
    {
        return InputFormat::json;
    }
    if (ends_with(norm, ".parquet"))
    {
        return InputFormat::parquet;
    }
    return InputFormat::unknown;
}

static void append_utf8(uint32_t cp, std::string &out)
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

bool next_codepoint(const std::string &s, std::size_t &i, uint32_t &cp)
{
    if (i >= s.size())
    {
        return false;
    }
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80)
    {
        cp = c;
        i += 1;
        return true;
    }
    if ((c >> 5) == 0x6 && i + 1 < s.size())
    {
        cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        i += 2;
        return true;
    }
    if ((c >> 4) == 0xE && i + 2 < s.size())
    {
        cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        i += 3;
        return true;
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size())
    {
        cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
             ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) | (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        i += 4;
        return true;
    }
    i += 1;
    cp = 0xFFFD;
    return true;
}

static bool is_space(uint32_t cp)
{
    if (cp <= 0x20)
    {
        return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D || cp == 0x20;
    }
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || cp == 0x180E || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

std::string truncate_utf8(const std::string &s, std::size_t max_chars)
{
    if (max_chars == 0)
    {
        return s;
    }
    std::size_t i = 0;
    std::size_t count = 0;
    while (i < s.size() && count < max_chars)
    {
        uint32_t cp = 0;
        std::size_t prev = i;
        if (!next_codepoint(s, i, cp))
        {
            break;
        }
        if (i == prev)
        {
            break;
        }
        ++count;
    }
    if (i >= s.size())
    {
        return s;
    }
    return s.substr(0, i);
}

std::vector<std::string> pretokenize(const std::string &text)
{
    std::vector<std::string> tokens;
    std::string pending_space;
    std::string current;
    std::size_t i = 0;
    while (i < text.size())
    {
        uint32_t cp = 0;
        std::size_t before = i;
        if (!next_codepoint(text, i, cp))
        {
            break;
        }
        if (i == before)
        {
            break;
        }
        std::string cp_utf8;
        append_utf8(cp, cp_utf8);
        if (is_space(cp))
        {
            if (!current.empty())
            {
                tokens.push_back(std::move(current));
                current.clear();
            }
            pending_space += cp_utf8;
        }
        else
        {
            if (current.empty())
            {
                current = pending_space;
                pending_space.clear();
            }
            current += cp_utf8;
        }
    }
    if (!current.empty())
    {
        tokens.push_back(std::move(current));
    }
    else if (!pending_space.empty())
    {
        tokens.push_back(std::move(pending_space));
    }
    return tokens;
}

std::array<uint32_t, 256> build_byte_to_unicode_cp()
{
    std::vector<int> bs;
    bs.reserve(256);
    std::array<bool, 256> present{};
    for (int b = 33; b <= 126; ++b)
    {
        bs.push_back(b);
        present[b] = true;
    }
    for (int b = 161; b <= 172; ++b)
    {
        bs.push_back(b);
        present[b] = true;
    }
    for (int b = 174; b <= 255; ++b)
    {
        bs.push_back(b);
        present[b] = true;
    }
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b)
    {
        if (!present[b])
        {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::array<uint32_t, 256> map{};
    for (std::size_t i = 0; i < bs.size(); ++i)
    {
        map[static_cast<std::size_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
    }
    return map;
}

std::array<std::string, 256> build_byte_to_unicode_str(const std::array<uint32_t, 256> &cp_map)
{
    std::array<std::string, 256> out;
    for (std::size_t i = 0; i < 256; ++i)
    {
        std::string s;
        append_utf8(cp_map[i], s);
        out[i] = std::move(s);
    }
    return out;
}

std::string byte_level_encode(const std::string &token, const std::array<std::string, 256> &byte_to_unicode)
{
    std::string out;
    out.reserve(token.size() * 2);
    for (unsigned char b : token)
    {
        out += byte_to_unicode[b];
    }
    return out;
}

static bool parse_hex4(const std::string &s, std::size_t i, uint32_t &out)
{
    if (i + 4 > s.size())
    {
        return false;
    }
    uint32_t val = 0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        char c = s[i + k];
        uint32_t v = 0;
        if (c >= '0' && c <= '9')
        {
            v = static_cast<uint32_t>(c - '0');
        }
        else if (c >= 'a' && c <= 'f')
        {
            v = static_cast<uint32_t>(10 + c - 'a');
        }
        else if (c >= 'A' && c <= 'F')
        {
            v = static_cast<uint32_t>(10 + c - 'A');
        }
        else
        {
            return false;
        }
        val = (val << 4) | v;
    }
    out = val;
    return true;
}

static bool parse_json_string(const std::string &s, std::size_t &i, std::string &out)
{
    if (i >= s.size() || s[i] != '"')
    {
        return false;
    }
    ++i;
    out.clear();
    while (i < s.size())
    {
        char c = s[i++];
        if (c == '"')
        {
            return true;
        }
        if (c == '\\')
        {
            if (i >= s.size())
            {
                return false;
            }
            char esc = s[i++];
            switch (esc)
            {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                uint32_t cp = 0;
                if (!parse_hex4(s, i, cp))
                {
                    return false;
                }
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF)
                {
                    if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u')
                    {
                        uint32_t low = 0;
                        if (parse_hex4(s, i + 2, low))
                        {
                            if (low >= 0xDC00 && low <= 0xDFFF)
                            {
                                i += 6;
                                cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                            }
                        }
                    }
                }
                append_utf8(cp, out);
                break;
            }
            default:
                out.push_back(esc);
                break;
            }
        }
        else
        {
            out.push_back(c);
        }
    }
    return false;
}

static void skip_ws(const std::string &s, std::size_t &i)
{
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
}

static bool skip_json_value(const std::string &s, std::size_t &i)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    if (s[i] == '"')
    {
        std::string tmp;
        return parse_json_string(s, i, tmp);
    }
    if (s[i] == '{' || s[i] == '[')
    {
        int depth = 0;
        while (i < s.size())
        {
            char c = s[i];
            if (c == '"')
            {
                std::string tmp;
                if (!parse_json_string(s, i, tmp))
                {
                    return false;
                }
                continue;
            }
            if (c == '{' || c == '[')
            {
                ++depth;
            }
            else if (c == '}' || c == ']')
            {
                --depth;
                if (depth == 0)
                {
                    ++i;
                    return true;
                }
            }
            ++i;
        }
        return false;
    }
    while (i < s.size())
    {
        char c = s[i];
        if (c == ',' || c == '}' || c == ']' || std::isspace(static_cast<unsigned char>(c)))
        {
            return true;
        }
        ++i;
    }
    return true;
}

bool extract_json_field(const std::string &s, const std::string &field, std::string &out)
{
    std::size_t i = 0;
    while (i < s.size())
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ',')
        {
            ++i;
            continue;
        }
        if (s[i] != '"')
        {
            ++i;
            continue;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            continue;
        }
        ++i;
        if (key == field)
        {
            skip_ws(s, i);
            if (i >= s.size() || s[i] != '"')
            {
                return false;
            }
            return parse_json_string(s, i, out);
        }
        if (!skip_json_value(s, i))
        {
            return false;
        }
    }
    return false;
}

std::unordered_map<std::string, uint32_t> reduce_top_k(std::unordered_map<std::string, uint32_t> &local,
                                                       std::size_t top_k)
{
    std::vector<std::pair<std::string, uint32_t>> vec;
    vec.reserve(local.size());
    for (auto &kv : local)
    {
        vec.emplace_back(std::move(kv.first), kv.second);
    }
    local.clear();
    if (vec.size() > top_k)
    {
        auto nth = vec.begin() + static_cast<std::ptrdiff_t>(top_k);
        std::nth_element(vec.begin(), nth, vec.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
        vec.resize(top_k);
    }
    std::unordered_map<std::string, uint32_t> reduced;
    reduced.reserve(vec.size() * 13 / 10 + 8);
    for (auto &kv : vec)
    {
        reduced.emplace(std::move(kv.first), kv.second);
    }
    return reduced;
}

bool read_gz_lines(const std::string &path, const std::function<void(const std::string &)> &cb)
{
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f)
    {
        return false;
    }
    const int buf_size = 1 << 20;
    std::string buf(buf_size, '\0');
    std::string line;
    while (true)
    {
        char *res = gzgets(f, buf.data(), buf_size);
        if (!res)
        {
            break;
        }
        line.assign(res);
        while (!line.empty() && line.back() != '\n' && !gzeof(f))
        {
            res = gzgets(f, buf.data(), buf_size);
            if (!res)
            {
                break;
            }
            line.append(res);
        }
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        {
            line.pop_back();
        }
        cb(line);
    }
    gzclose(f);
    return true;
}

bool read_text_lines(const std::string &path, const std::function<void(const std::string &)> &cb)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        cb(line);
    }
    return true;
}

static bool decode_xz_stream(std::istream &in, const std::function<void(const char *, std::size_t)> &on_chunk)
{
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_stream_decoder(&strm, UINT64_MAX, 0) != LZMA_OK)
    {
        return false;
    }

    std::vector<std::uint8_t> in_buf(1 << 16);
    std::vector<std::uint8_t> out_buf(1 << 16);
    lzma_action action = LZMA_RUN;
    bool eof = false;
    bool ok = true;

    while (true)
    {
        if (strm.avail_in == 0 && !eof)
        {
            in.read(reinterpret_cast<char *>(in_buf.data()), static_cast<std::streamsize>(in_buf.size()));
            std::streamsize got = in.gcount();
            if (got < 0)
            {
                ok = false;
                break;
            }
            strm.next_in = in_buf.data();
            strm.avail_in = static_cast<std::size_t>(got);
            if (got == 0)
            {
                eof = true;
                action = LZMA_FINISH;
            }
        }

        strm.next_out = out_buf.data();
        strm.avail_out = out_buf.size();

        lzma_ret ret = lzma_code(&strm, action);
        std::size_t produced = out_buf.size() - strm.avail_out;
        if (produced > 0)
        {
            on_chunk(reinterpret_cast<const char *>(out_buf.data()), produced);
        }

        if (ret == LZMA_STREAM_END)
        {
            break;
        }
        if (ret != LZMA_OK)
        {
            ok = false;
            break;
        }
        if (eof && strm.avail_in == 0 && produced == 0)
        {
            ok = false;
            break;
        }
    }

    lzma_end(&strm);
    return ok;
}

bool read_xz_lines(const std::string &path, const std::function<void(const std::string &)> &cb)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }

    std::string pending;
    bool ok = decode_xz_stream(in, [&](const char *data, std::size_t n) {
        pending.append(data, n);
        std::size_t start = 0;
        while (start < pending.size())
        {
            std::size_t pos = pending.find('\n', start);
            if (pos == std::string::npos)
            {
                break;
            }
            std::string line = pending.substr(start, pos - start);
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            cb(line);
            start = pos + 1;
        }
        if (start > 0)
        {
            pending.erase(0, start);
        }
    });
    if (!ok)
    {
        return false;
    }

    if (!pending.empty())
    {
        if (!pending.empty() && pending.back() == '\r')
        {
            pending.pop_back();
        }
        cb(pending);
    }
    return true;
}

static bool read_text_file_all(const std::string &path, std::string &out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return static_cast<bool>(in) || in.eof();
}

static bool read_gz_file_all(const std::string &path, std::string &out)
{
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f)
    {
        return false;
    }
    std::vector<char> buf(1 << 16);
    out.clear();
    while (true)
    {
        int got = gzread(f, buf.data(), static_cast<unsigned int>(buf.size()));
        if (got < 0)
        {
            gzclose(f);
            return false;
        }
        if (got == 0)
        {
            break;
        }
        out.append(buf.data(), static_cast<std::size_t>(got));
    }
    gzclose(f);
    return true;
}

static bool read_xz_file_all(const std::string &path, std::string &out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    out.clear();
    return decode_xz_stream(in, [&](const char *data, std::size_t n) { out.append(data, n); });
}

static bool parse_json_object_for_field(const std::string &s, std::size_t &i, const std::string &field,
                                        const std::function<void(const std::string &)> &cb)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        skip_ws(s, i);
        if (key == field && i < s.size() && s[i] == '"')
        {
            std::string text;
            if (!parse_json_string(s, i, text))
            {
                return false;
            }
            if (!text.empty())
            {
                cb(text);
            }
        }
        else
        {
            if (!skip_json_value(s, i))
            {
                return false;
            }
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_json_array_for_field(const std::string &s, std::size_t &i, const std::string &field,
                                       const std::function<void(const std::string &)> &cb)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] == '{')
        {
            if (!parse_json_object_for_field(s, i, field, cb))
            {
                return false;
            }
        }
        else
        {
            if (!skip_json_value(s, i))
            {
                return false;
            }
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool for_each_text_record_in_json_document(const std::string &payload, const std::string &text_field,
                                                  const std::function<void(const std::string &)> &cb)
{
    std::size_t i = 0;
    skip_ws(payload, i);
    if (i >= payload.size())
    {
        return true;
    }
    if (payload[i] == '[')
    {
        return parse_json_array_for_field(payload, i, text_field, cb);
    }
    if (payload[i] == '{')
    {
        return parse_json_object_for_field(payload, i, text_field, cb);
    }
    return false;
}

static bool read_parquet_records(const std::string &path, const std::string &text_field,
                                 const std::function<void(const std::string &)> &cb, std::string &err);

static bool for_each_text_record_in_lines(const std::function<bool(const std::function<void(const std::string &)> &)> &reader,
                                          const std::string &text_field, const std::function<void(const std::string &)> &cb,
                                          std::size_t &matched)
{
    matched = 0;
    return reader([&](const std::string &line) {
        if (line.empty())
        {
            return;
        }
        std::string text;
        if (!extract_json_field(line, text_field, text) || text.empty())
        {
            return;
        }
        cb(text);
        ++matched;
    });
}

bool for_each_text_record(const std::string &path, const std::string &text_field,
                          const std::function<void(const std::string &)> &cb, std::string &err)
{
    InputFormat fmt = detect_input_format(path);
    std::size_t matched = 0;

    switch (fmt)
    {
    case InputFormat::jsonl:
        if (!for_each_text_record_in_lines([&](const auto &line_cb) { return read_text_lines(path, line_cb); }, text_field,
                                           cb, matched))
        {
            err = "failed to read text lines: " + path;
            return false;
        }
        return true;
    case InputFormat::jsonl_gz:
        if (!for_each_text_record_in_lines([&](const auto &line_cb) { return read_gz_lines(path, line_cb); }, text_field,
                                           cb, matched))
        {
            err = "failed to read gz lines: " + path;
            return false;
        }
        return true;
    case InputFormat::jsonl_xz:
        if (!for_each_text_record_in_lines([&](const auto &line_cb) { return read_xz_lines(path, line_cb); }, text_field,
                                           cb, matched))
        {
            err = "failed to read xz lines: " + path;
            return false;
        }
        return true;
    case InputFormat::json:
    case InputFormat::json_gz:
    case InputFormat::json_xz: {
        bool line_ok = false;
        if (fmt == InputFormat::json)
        {
            line_ok = for_each_text_record_in_lines([&](const auto &line_cb) { return read_text_lines(path, line_cb); },
                                                    text_field, cb, matched);
        }
        else if (fmt == InputFormat::json_gz)
        {
            line_ok = for_each_text_record_in_lines([&](const auto &line_cb) { return read_gz_lines(path, line_cb); },
                                                    text_field, cb, matched);
        }
        else
        {
            line_ok = for_each_text_record_in_lines([&](const auto &line_cb) { return read_xz_lines(path, line_cb); },
                                                    text_field, cb, matched);
        }
        if (!line_ok)
        {
            err = "failed to read json stream: " + path;
            return false;
        }
        if (matched > 0)
        {
            return true;
        }

        std::string payload;
        bool read_ok = false;
        if (fmt == InputFormat::json)
        {
            read_ok = read_text_file_all(path, payload);
        }
        else if (fmt == InputFormat::json_gz)
        {
            read_ok = read_gz_file_all(path, payload);
        }
        else
        {
            read_ok = read_xz_file_all(path, payload);
        }
        if (!read_ok)
        {
            err = "failed to read full json document: " + path;
            return false;
        }
        if (!for_each_text_record_in_json_document(payload, text_field, cb))
        {
            err = "failed to parse json document: " + path;
            return false;
        }
        return true;
    }
    case InputFormat::parquet:
        return read_parquet_records(path, text_field, cb, err);
    case InputFormat::unknown:
    default:
        // Fallback: try line-based json extraction for arbitrary extensions.
        if (!for_each_text_record_in_lines([&](const auto &line_cb) { return read_text_lines(path, line_cb); }, text_field,
                                           cb, matched))
        {
            err = "unsupported input format: " + path;
            return false;
        }
        return true;
    }
}

static std::string json_escape(const std::string &s)
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

bool write_vocab_json(const std::string &path, const std::vector<std::string> &id_to_token)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << "{";
    for (std::size_t i = 0; i < id_to_token.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "\"" << json_escape(id_to_token[i]) << "\":" << i;
    }
    out << "}";
    return true;
}

bool write_merges_txt(const std::string &path, const std::vector<std::string> &merges_out)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << "#version: 0.2\n";
    for (const auto &m : merges_out)
    {
        out << m << "\n";
    }
    return true;
}

bool write_tokenizer_json(const std::string &path, const Config &cfg, const std::vector<std::string> &id_to_token,
                          const std::vector<std::string> &merges_out)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << "{";
    out << "\"version\":\"1.0\",";
    out << "\"truncation\":null,";
    out << "\"padding\":null,";

    out << "\"added_tokens\":[";
    for (std::size_t i = 0; i < cfg.special_tokens.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "{";
        out << "\"id\":" << i << ",";
        out << "\"content\":\"" << json_escape(cfg.special_tokens[i]) << "\",";
        out << "\"single_word\":false,";
        out << "\"lstrip\":false,";
        out << "\"rstrip\":false,";
        out << "\"normalized\":false,";
        out << "\"special\":true";
        out << "}";
    }
    out << "],";

    out << "\"normalizer\":null,";
    out << "\"pre_tokenizer\":{";
    out << "\"type\":\"ByteLevel\",";
    out << "\"add_prefix_space\":false,";
    out << "\"trim_offsets\":true";
    out << "},";
    out << "\"post_processor\":null,";
    out << "\"decoder\":{";
    out << "\"type\":\"ByteLevel\",";
    out << "\"add_prefix_space\":false,";
    out << "\"trim_offsets\":true";
    out << "},";

    out << "\"model\":{";
    out << "\"type\":\"BPE\",";
    out << "\"dropout\":null,";
    out << "\"unk_token\":\"" << json_escape(cfg.unk_token) << "\",";
    out << "\"continuing_subword_prefix\":\"\",";
    out << "\"end_of_word_suffix\":\"\",";
    out << "\"fuse_unk\":false,";

    out << "\"vocab\":{";
    for (std::size_t i = 0; i < id_to_token.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "\"" << json_escape(id_to_token[i]) << "\":" << i;
    }
    out << "},";

    out << "\"merges\":[";
    for (std::size_t i = 0; i < merges_out.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "\"" << json_escape(merges_out[i]) << "\"";
    }
    out << "]";
    out << "}";
    out << "}";
    return true;
}

bool write_chunk_file(const std::string &path, const std::unordered_map<std::string, uint32_t> &counts,
                      uint64_t doc_count)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    std::vector<char> payload;
    payload.reserve(counts.size() * 32 + 16);
    auto append_bytes = [&](const void *ptr, std::size_t n) {
        const char *p = reinterpret_cast<const char *>(ptr);
        payload.insert(payload.end(), p, p + n);
    };
    for (const auto &kv : counts)
    {
        uint32_t len = static_cast<uint32_t>(kv.first.size());
        append_bytes(&len, sizeof(len));
        append_bytes(kv.first.data(), len);
        uint32_t count = kv.second;
        append_bytes(&count, sizeof(count));
    }

    uLongf dst_bound = compressBound(static_cast<uLong>(payload.size()));
    std::vector<Bytef> compressed(dst_bound);
    uLongf compressed_size = dst_bound;
    int zret = compress2(compressed.data(), &compressed_size, reinterpret_cast<const Bytef *>(payload.data()),
                         static_cast<uLong>(payload.size()), Z_BEST_SPEED);
    if (zret != Z_OK)
    {
        return false;
    }

    ChunkHeader header;
    header.doc_count = doc_count;
    header.entry_count = counts.size();
    header.payload_uncompressed_bytes = static_cast<uint64_t>(payload.size());
    header.payload_compressed_bytes = static_cast<uint64_t>(compressed_size);
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!out)
    {
        return false;
    }

    out.write(reinterpret_cast<const char *>(compressed.data()), static_cast<std::streamsize>(compressed_size));
    return true;
}

bool read_chunk_header(const std::string &path, ChunkHeader &header)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    struct LegacyHeader
    {
        uint32_t magic = 0;
        uint32_t version = 0;
        uint64_t doc_count = 0;
        uint64_t entry_count = 0;
    };
    LegacyHeader legacy;
    in.read(reinterpret_cast<char *>(&legacy), sizeof(legacy));
    if (!in)
    {
        return false;
    }

    if (legacy.magic == 0x314B4243 && legacy.version == 1)
    {
        header.magic = legacy.magic;
        header.version = legacy.version;
        header.doc_count = legacy.doc_count;
        header.entry_count = legacy.entry_count;
        header.payload_uncompressed_bytes = 0;
        header.payload_compressed_bytes = 0;
        return true;
    }

    if (legacy.magic == 0x324B4243 && legacy.version == 2)
    {
        header.magic = legacy.magic;
        header.version = legacy.version;
        header.doc_count = legacy.doc_count;
        header.entry_count = legacy.entry_count;
        in.read(reinterpret_cast<char *>(&header.payload_uncompressed_bytes), sizeof(header.payload_uncompressed_bytes));
        in.read(reinterpret_cast<char *>(&header.payload_compressed_bytes), sizeof(header.payload_compressed_bytes));
        if (!in)
        {
            return false;
        }
        return true;
    }

    return false;
}

bool merge_chunk_file(const std::string &path, std::unordered_map<std::string, uint64_t> &global_counts,
                      uint64_t *doc_count_out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    struct LegacyHeader
    {
        uint32_t magic = 0;
        uint32_t version = 0;
        uint64_t doc_count = 0;
        uint64_t entry_count = 0;
    };
    LegacyHeader header;
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in)
    {
        return false;
    }
    if (doc_count_out)
    {
        *doc_count_out += header.doc_count;
    }

    if (header.magic == 0x314B4243 && header.version == 1)
    {
        for (uint64_t i = 0; i < header.entry_count; ++i)
        {
            uint32_t len = 0;
            in.read(reinterpret_cast<char *>(&len), sizeof(len));
            if (!in)
            {
                return false;
            }
            std::string key(len, '\0');
            if (len > 0)
            {
                in.read(&key[0], len);
                if (!in)
                {
                    return false;
                }
            }
            uint32_t count = 0;
            in.read(reinterpret_cast<char *>(&count), sizeof(count));
            if (!in)
            {
                return false;
            }
            global_counts[key] += count;
        }
        return true;
    }

    if (header.magic != 0x324B4243 || header.version != 2)
    {
        return false;
    }

    uint64_t raw_bytes = 0;
    uint64_t compressed_bytes = 0;
    in.read(reinterpret_cast<char *>(&raw_bytes), sizeof(raw_bytes));
    in.read(reinterpret_cast<char *>(&compressed_bytes), sizeof(compressed_bytes));
    if (!in)
    {
        return false;
    }
    if (compressed_bytes == 0 && raw_bytes > 0)
    {
        return false;
    }

    std::vector<Bytef> compressed(static_cast<std::size_t>(compressed_bytes));
    if (compressed_bytes > 0)
    {
        in.read(reinterpret_cast<char *>(compressed.data()), static_cast<std::streamsize>(compressed_bytes));
        if (!in)
        {
            return false;
        }
    }

    std::vector<char> payload(static_cast<std::size_t>(raw_bytes));
    if (raw_bytes > 0)
    {
        uLongf out_size = static_cast<uLongf>(raw_bytes);
        int zret = uncompress(reinterpret_cast<Bytef *>(payload.data()), &out_size, compressed.data(),
                              static_cast<uLong>(compressed.size()));
        if (zret != Z_OK || out_size != raw_bytes)
        {
            return false;
        }
    }

    std::size_t pos = 0;
    auto read_u32 = [&](uint32_t &x) -> bool {
        if (pos + sizeof(uint32_t) > payload.size())
        {
            return false;
        }
        std::memcpy(&x, payload.data() + pos, sizeof(uint32_t));
        pos += sizeof(uint32_t);
        return true;
    };

    for (uint64_t i = 0; i < header.entry_count; ++i)
    {
        uint32_t len = 0;
        if (!read_u32(len))
        {
            return false;
        }
        if (pos + len > payload.size())
        {
            return false;
        }
        std::string key(payload.data() + pos, payload.data() + pos + len);
        pos += len;
        uint32_t count = 0;
        if (!read_u32(count))
        {
            return false;
        }
        global_counts[key] += count;
    }
    return pos == payload.size();
}

static std::string format_duration(double seconds)
{
    int sec = static_cast<int>(seconds + 0.5);
    int h = sec / 3600;
    int m = (sec % 3600) / 60;
    int s = sec % 60;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
    return oss.str();
}

ProgressTracker::ProgressTracker(uint64_t total_chunks, const std::string &label, uint64_t interval_ms)
    : label_(label), total_(total_chunks), interval_ms_(interval_ms)
{
    start_ = std::chrono::steady_clock::now();
    last_print_ = start_;
}

void ProgressTracker::add_total(uint64_t chunks)
{
    total_.fetch_add(chunks, std::memory_order_relaxed);
    maybe_print(false);
}

void ProgressTracker::set_total(uint64_t chunks)
{
    total_.store(chunks, std::memory_order_relaxed);
    maybe_print(false);
}

void ProgressTracker::add(uint64_t chunks, uint64_t docs)
{
    done_chunks_.fetch_add(chunks, std::memory_order_relaxed);
    done_docs_.fetch_add(docs, std::memory_order_relaxed);
    maybe_print(false);
}

void ProgressTracker::finish()
{
    maybe_print(true);
}

void ProgressTracker::maybe_print(bool force)
{
    auto now = std::chrono::steady_clock::now();
    if (!force)
    {
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_).count();
        if (delta < static_cast<long long>(interval_ms_))
        {
            return;
        }
    }
    std::lock_guard<std::mutex> lock(print_mu_);
    now = std::chrono::steady_clock::now();
    if (!force)
    {
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_).count();
        if (delta < static_cast<long long>(interval_ms_))
        {
            return;
        }
    }
    last_print_ = now;
    uint64_t total_chunks = total_.load(std::memory_order_relaxed);
    uint64_t done_chunks = done_chunks_.load(std::memory_order_relaxed);
    uint64_t done_docs = done_docs_.load(std::memory_order_relaxed);
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_).count();
    double chunk_rate = elapsed > 0.0 ? static_cast<double>(done_chunks) / elapsed : 0.0;
    double doc_rate = elapsed > 0.0 ? static_cast<double>(done_docs) / elapsed : 0.0;
    double pct = 0.0;
    if (total_chunks > 0)
    {
        pct = 100.0 * static_cast<double>(done_chunks) / static_cast<double>(total_chunks);
    }
    double eta =
        (chunk_rate > 0.0 && total_chunks > done_chunks) ? static_cast<double>(total_chunks - done_chunks) / chunk_rate
                                                          : 0.0;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    if (total_chunks > 0)
    {
        oss << "[" << label_ << "] chunks " << done_chunks << "/" << total_chunks << " (" << std::setprecision(1)
            << pct << "%)";
    }
    else
    {
        oss << "[" << label_ << "] chunks " << done_chunks;
    }
    if (done_docs > 0)
    {
        oss << " docs " << done_docs << " (" << std::setprecision(1) << (static_cast<double>(done_docs) / 1e6) << "M)";
    }
    if (chunk_rate > 0.0)
    {
        oss << " rate " << std::setprecision(2) << chunk_rate << " c/s";
    }
    if (doc_rate > 0.0)
    {
        oss << " " << std::setprecision(2) << (doc_rate / 1000.0) << " kdoc/s";
    }
    if (eta > 0.0)
    {
        oss << " ETA " << format_duration(eta);
    }
    oss << "\n";
    std::cerr << oss.str();
}

namespace
{
namespace pq
{
constexpr uint8_t CT_STOP = 0x00;
constexpr uint8_t CT_BOOLEAN_TRUE = 0x01;
constexpr uint8_t CT_BOOLEAN_FALSE = 0x02;
constexpr uint8_t CT_BYTE = 0x03;
constexpr uint8_t CT_I16 = 0x04;
constexpr uint8_t CT_I32 = 0x05;
constexpr uint8_t CT_I64 = 0x06;
constexpr uint8_t CT_BINARY = 0x08;
constexpr uint8_t CT_LIST = 0x09;
constexpr uint8_t CT_SET = 0x0A;
constexpr uint8_t CT_MAP = 0x0B;
constexpr uint8_t CT_STRUCT = 0x0C;

constexpr int32_t REP_REQUIRED = 0;
constexpr int32_t REP_OPTIONAL = 1;
constexpr int32_t REP_REPEATED = 2;

constexpr int32_t TYPE_BYTE_ARRAY = 6;

constexpr int32_t CODEC_UNCOMPRESSED = 0;
constexpr int32_t CODEC_SNAPPY = 1;
constexpr int32_t CODEC_GZIP = 2;

constexpr int32_t ENC_PLAIN = 0;
constexpr int32_t ENC_PLAIN_DICTIONARY = 2;
constexpr int32_t ENC_RLE = 3;
constexpr int32_t ENC_RLE_DICTIONARY = 8;

constexpr int32_t PAGE_DATA = 0;
constexpr int32_t PAGE_DICTIONARY = 2;
constexpr int32_t PAGE_DATA_V2 = 3;

struct SchemaElement
{
    int32_t type = -1;
    int32_t repetition_type = REP_REQUIRED;
    int32_t num_children = 0;
    std::string name;
};

struct ColumnMetaData
{
    int32_t type = -1;
    int32_t codec = CODEC_UNCOMPRESSED;
    std::vector<std::string> path_in_schema;
    int64_t num_values = 0;
    int64_t total_compressed_size = 0;
    int64_t total_uncompressed_size = 0;
    int64_t data_page_offset = -1;
    int64_t dictionary_page_offset = -1;
    int64_t file_offset = -1;
};

struct RowGroupMetaData
{
    int64_t num_rows = 0;
    std::vector<ColumnMetaData> columns;
};

struct FileMetaData
{
    int64_t num_rows = 0;
    std::vector<SchemaElement> schema;
    std::vector<RowGroupMetaData> row_groups;
};

struct DataPageHeader
{
    int32_t num_values = 0;
    int32_t encoding = ENC_PLAIN;
    int32_t definition_level_encoding = ENC_RLE;
    int32_t repetition_level_encoding = ENC_RLE;
};

struct DataPageHeaderV2
{
    int32_t num_values = 0;
    int32_t num_nulls = 0;
    int32_t num_rows = 0;
    int32_t encoding = ENC_PLAIN;
    int32_t definition_levels_byte_length = 0;
    int32_t repetition_levels_byte_length = 0;
    bool is_compressed = true;
};

struct DictionaryPageHeader
{
    int32_t num_values = 0;
    int32_t encoding = ENC_PLAIN;
    bool is_sorted = false;
};

struct PageHeader
{
    int32_t type = -1;
    int32_t uncompressed_page_size = 0;
    int32_t compressed_page_size = 0;
    bool has_data_page_header = false;
    bool has_dictionary_page_header = false;
    bool has_data_page_header_v2 = false;
    DataPageHeader data_page_header;
    DataPageHeaderV2 data_page_header_v2;
    DictionaryPageHeader dictionary_page_header;
};

class BufferReader
{
  public:
    BufferReader(const uint8_t *data, std::size_t size) : data_(data), size_(size) {}

    bool read_u8(uint8_t &out)
    {
        if (pos_ >= size_)
        {
            return false;
        }
        out = data_[pos_++];
        return true;
    }

    bool read_exact(uint8_t *dst, std::size_t n)
    {
        if (pos_ + n > size_)
        {
            return false;
        }
        std::memcpy(dst, data_ + pos_, n);
        pos_ += n;
        return true;
    }

    bool skip(std::size_t n)
    {
        if (pos_ + n > size_)
        {
            return false;
        }
        pos_ += n;
        return true;
    }

    std::size_t pos() const { return pos_; }
    std::size_t remaining() const { return size_ - pos_; }

  private:
    const uint8_t *data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t pos_ = 0;
};

class StreamReader
{
  public:
    StreamReader(std::ifstream &in, uint64_t start, uint64_t limit) : in_(in), pos_(start), limit_(limit) {}

    bool read_u8(uint8_t &out)
    {
        if (pos_ >= limit_)
        {
            return false;
        }
        char c = 0;
        in_.read(&c, 1);
        if (!in_)
        {
            return false;
        }
        out = static_cast<uint8_t>(c);
        ++pos_;
        return true;
    }

    bool read_exact(uint8_t *dst, std::size_t n)
    {
        if (n == 0)
        {
            return true;
        }
        if (pos_ + n > limit_)
        {
            return false;
        }
        in_.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(n));
        if (!in_)
        {
            return false;
        }
        pos_ += n;
        return true;
    }

    bool skip(std::size_t n)
    {
        if (n == 0)
        {
            return true;
        }
        if (pos_ + n > limit_)
        {
            return false;
        }
        in_.seekg(static_cast<std::streamoff>(n), std::ios::cur);
        if (!in_)
        {
            return false;
        }
        pos_ += n;
        return true;
    }

    uint64_t pos() const { return pos_; }

  private:
    std::ifstream &in_;
    uint64_t pos_ = 0;
    uint64_t limit_ = 0;
};

template <typename Reader>
static bool read_varint(Reader &r, uint64_t &out)
{
    out = 0;
    int shift = 0;
    for (int i = 0; i < 10; ++i)
    {
        uint8_t b = 0;
        if (!r.read_u8(b))
        {
            return false;
        }
        out |= static_cast<uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0)
        {
            return true;
        }
        shift += 7;
    }
    return false;
}

static int64_t zigzag_decode(uint64_t n)
{
    return static_cast<int64_t>((n >> 1) ^ static_cast<uint64_t>(-static_cast<int64_t>(n & 1)));
}

template <typename Reader>
static bool read_i32(Reader &r, int32_t &out)
{
    uint64_t raw = 0;
    if (!read_varint(r, raw))
    {
        return false;
    }
    out = static_cast<int32_t>(zigzag_decode(raw));
    return true;
}

template <typename Reader>
static bool read_i16(Reader &r, int16_t &out)
{
    int32_t tmp = 0;
    if (!read_i32(r, tmp))
    {
        return false;
    }
    out = static_cast<int16_t>(tmp);
    return true;
}

template <typename Reader>
static bool read_i64(Reader &r, int64_t &out)
{
    uint64_t raw = 0;
    if (!read_varint(r, raw))
    {
        return false;
    }
    out = zigzag_decode(raw);
    return true;
}

template <typename Reader>
static bool read_binary(Reader &r, std::string &out)
{
    uint64_t len = 0;
    if (!read_varint(r, len))
    {
        return false;
    }
    if (len > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        return false;
    }
    out.resize(static_cast<std::size_t>(len));
    if (len == 0)
    {
        return true;
    }
    return r.read_exact(reinterpret_cast<uint8_t *>(out.data()), static_cast<std::size_t>(len));
}

template <typename Reader>
static bool read_bool(Reader &r, bool bool_inline, bool inline_value, bool &out)
{
    if (bool_inline)
    {
        out = inline_value;
        return true;
    }
    uint8_t b = 0;
    if (!r.read_u8(b))
    {
        return false;
    }
    if (b == CT_BOOLEAN_TRUE || b == 1)
    {
        out = true;
        return true;
    }
    if (b == CT_BOOLEAN_FALSE || b == 2 || b == 0)
    {
        out = false;
        return true;
    }
    return false;
}

template <typename Reader>
static bool read_field_header(Reader &r, int16_t &last_id, int16_t &field_id, uint8_t &ctype, bool &bool_inline,
                              bool &bool_value)
{
    uint8_t b = 0;
    if (!r.read_u8(b))
    {
        return false;
    }
    if (b == CT_STOP)
    {
        field_id = 0;
        ctype = CT_STOP;
        bool_inline = false;
        bool_value = false;
        return true;
    }

    uint8_t delta = static_cast<uint8_t>(b >> 4);
    ctype = static_cast<uint8_t>(b & 0x0F);
    if (delta == 0)
    {
        if (!read_i16(r, field_id))
        {
            return false;
        }
    }
    else
    {
        field_id = static_cast<int16_t>(last_id + delta);
    }
    last_id = field_id;
    bool_inline = (ctype == CT_BOOLEAN_TRUE || ctype == CT_BOOLEAN_FALSE);
    bool_value = (ctype == CT_BOOLEAN_TRUE);
    return true;
}

template <typename Reader>
static bool read_list_header(Reader &r, uint64_t &size, uint8_t &elem_type)
{
    uint8_t b = 0;
    if (!r.read_u8(b))
    {
        return false;
    }
    elem_type = static_cast<uint8_t>(b & 0x0F);
    uint8_t sz = static_cast<uint8_t>(b >> 4);
    if (sz == 15)
    {
        return read_varint(r, size);
    }
    size = sz;
    return true;
}

template <typename Reader>
static bool read_map_header(Reader &r, uint64_t &size, uint8_t &key_type, uint8_t &val_type)
{
    if (!read_varint(r, size))
    {
        return false;
    }
    if (size == 0)
    {
        key_type = CT_STOP;
        val_type = CT_STOP;
        return true;
    }
    uint8_t kv = 0;
    if (!r.read_u8(kv))
    {
        return false;
    }
    key_type = static_cast<uint8_t>(kv >> 4);
    val_type = static_cast<uint8_t>(kv & 0x0F);
    return true;
}

template <typename Reader>
static bool skip_value(Reader &r, uint8_t ctype, bool bool_inline);

template <typename Reader>
static bool skip_struct(Reader &r)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        if (!skip_value(r, ctype, bool_inline))
        {
            return false;
        }
    }
}

template <typename Reader>
static bool skip_value(Reader &r, uint8_t ctype, bool bool_inline)
{
    switch (ctype)
    {
    case CT_STOP:
        return true;
    case CT_BOOLEAN_TRUE:
        return true;
    case CT_BOOLEAN_FALSE:
        if (bool_inline)
        {
            return true;
        }
        return r.skip(1);
    case CT_BYTE:
        return r.skip(1);
    case CT_I16: {
        int16_t v = 0;
        return read_i16(r, v);
    }
    case CT_I32: {
        int32_t v = 0;
        return read_i32(r, v);
    }
    case CT_I64: {
        int64_t v = 0;
        return read_i64(r, v);
    }
    case CT_BINARY: {
        uint64_t len = 0;
        if (!read_varint(r, len))
        {
            return false;
        }
        if (len > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            return false;
        }
        return r.skip(static_cast<std::size_t>(len));
    }
    case CT_LIST:
    case CT_SET: {
        uint64_t n = 0;
        uint8_t elem = CT_STOP;
        if (!read_list_header(r, n, elem))
        {
            return false;
        }
        for (uint64_t i = 0; i < n; ++i)
        {
            if (!skip_value(r, elem, false))
            {
                return false;
            }
        }
        return true;
    }
    case CT_MAP: {
        uint64_t n = 0;
        uint8_t key = CT_STOP;
        uint8_t val = CT_STOP;
        if (!read_map_header(r, n, key, val))
        {
            return false;
        }
        for (uint64_t i = 0; i < n; ++i)
        {
            if (!skip_value(r, key, false) || !skip_value(r, val, false))
            {
                return false;
            }
        }
        return true;
    }
    case CT_STRUCT:
        return skip_struct(r);
    default:
        return false;
    }
}

static bool parse_schema_element(BufferReader &r, SchemaElement &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 1:
            if (ctype != CT_I32 || !read_i32(r, out.type))
            {
                return false;
            }
            break;
        case 3:
            if (ctype != CT_I32 || !read_i32(r, out.repetition_type))
            {
                return false;
            }
            break;
        case 4:
            if (ctype != CT_BINARY || !read_binary(r, out.name))
            {
                return false;
            }
            break;
        case 5:
            if (ctype != CT_I32 || !read_i32(r, out.num_children))
            {
                return false;
            }
            break;
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

static bool parse_column_metadata(BufferReader &r, ColumnMetaData &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 1:
            if (ctype != CT_I32 || !read_i32(r, out.type))
            {
                return false;
            }
            break;
        case 3:
            if (ctype != CT_LIST)
            {
                return false;
            }
            else
            {
                uint64_t n = 0;
                uint8_t elem = CT_STOP;
                if (!read_list_header(r, n, elem))
                {
                    return false;
                }
                out.path_in_schema.clear();
                out.path_in_schema.reserve(static_cast<std::size_t>(n));
                for (uint64_t i = 0; i < n; ++i)
                {
                    if (elem == CT_BINARY)
                    {
                        std::string s;
                        if (!read_binary(r, s))
                        {
                            return false;
                        }
                        out.path_in_schema.push_back(std::move(s));
                    }
                    else
                    {
                        if (!skip_value(r, elem, false))
                        {
                            return false;
                        }
                    }
                }
            }
            break;
        case 4:
            if (ctype != CT_I32 || !read_i32(r, out.codec))
            {
                return false;
            }
            break;
        case 5:
            if (ctype != CT_I64 || !read_i64(r, out.num_values))
            {
                return false;
            }
            break;
        case 6:
            if (ctype != CT_I64 || !read_i64(r, out.total_uncompressed_size))
            {
                return false;
            }
            break;
        case 7:
            if (ctype != CT_I64 || !read_i64(r, out.total_compressed_size))
            {
                return false;
            }
            break;
        case 9:
            if (ctype != CT_I64 || !read_i64(r, out.data_page_offset))
            {
                return false;
            }
            break;
        case 11:
            if (ctype != CT_I64 || !read_i64(r, out.dictionary_page_offset))
            {
                return false;
            }
            break;
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

static bool parse_column_chunk(BufferReader &r, ColumnMetaData &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 2:
            if (ctype != CT_I64 || !read_i64(r, out.file_offset))
            {
                return false;
            }
            break;
        case 3:
            if (ctype != CT_STRUCT || !parse_column_metadata(r, out))
            {
                return false;
            }
            break;
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

static bool parse_row_group(BufferReader &r, RowGroupMetaData &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 1:
            if (ctype != CT_LIST)
            {
                return false;
            }
            else
            {
                uint64_t n = 0;
                uint8_t elem = CT_STOP;
                if (!read_list_header(r, n, elem))
                {
                    return false;
                }
                out.columns.clear();
                out.columns.reserve(static_cast<std::size_t>(n));
                for (uint64_t i = 0; i < n; ++i)
                {
                    if (elem == CT_STRUCT)
                    {
                        ColumnMetaData col;
                        if (!parse_column_chunk(r, col))
                        {
                            return false;
                        }
                        out.columns.push_back(std::move(col));
                    }
                    else
                    {
                        if (!skip_value(r, elem, false))
                        {
                            return false;
                        }
                    }
                }
            }
            break;
        case 3:
            if (ctype != CT_I64 || !read_i64(r, out.num_rows))
            {
                return false;
            }
            break;
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

static bool parse_file_metadata(BufferReader &r, FileMetaData &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 2:
            if (ctype != CT_LIST)
            {
                return false;
            }
            else
            {
                uint64_t n = 0;
                uint8_t elem = CT_STOP;
                if (!read_list_header(r, n, elem))
                {
                    return false;
                }
                out.schema.clear();
                out.schema.reserve(static_cast<std::size_t>(n));
                for (uint64_t i = 0; i < n; ++i)
                {
                    if (elem == CT_STRUCT)
                    {
                        SchemaElement e;
                        if (!parse_schema_element(r, e))
                        {
                            return false;
                        }
                        out.schema.push_back(std::move(e));
                    }
                    else
                    {
                        if (!skip_value(r, elem, false))
                        {
                            return false;
                        }
                    }
                }
            }
            break;
        case 3:
            if (ctype != CT_I64 || !read_i64(r, out.num_rows))
            {
                return false;
            }
            break;
        case 4:
            if (ctype != CT_LIST)
            {
                return false;
            }
            else
            {
                uint64_t n = 0;
                uint8_t elem = CT_STOP;
                if (!read_list_header(r, n, elem))
                {
                    return false;
                }
                out.row_groups.clear();
                out.row_groups.reserve(static_cast<std::size_t>(n));
                for (uint64_t i = 0; i < n; ++i)
                {
                    if (elem == CT_STRUCT)
                    {
                        RowGroupMetaData rg;
                        if (!parse_row_group(r, rg))
                        {
                            return false;
                        }
                        out.row_groups.push_back(std::move(rg));
                    }
                    else
                    {
                        if (!skip_value(r, elem, false))
                        {
                            return false;
                        }
                    }
                }
            }
            break;
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

template <typename Reader>
static bool parse_data_page_header(Reader &r, DataPageHeader &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 1:
            if (ctype != CT_I32 || !read_i32(r, out.num_values))
            {
                return false;
            }
            break;
        case 2:
            if (ctype != CT_I32 || !read_i32(r, out.encoding))
            {
                return false;
            }
            break;
        case 3:
            if (ctype != CT_I32 || !read_i32(r, out.definition_level_encoding))
            {
                return false;
            }
            break;
        case 4:
            if (ctype != CT_I32 || !read_i32(r, out.repetition_level_encoding))
            {
                return false;
            }
            break;
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

template <typename Reader>
static bool parse_dictionary_page_header(Reader &r, DictionaryPageHeader &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 1:
            if (ctype != CT_I32 || !read_i32(r, out.num_values))
            {
                return false;
            }
            break;
        case 2:
            if (ctype != CT_I32 || !read_i32(r, out.encoding))
            {
                return false;
            }
            break;
        case 3: {
            bool v = false;
            if (!read_bool(r, bool_inline, bool_value, v))
            {
                return false;
            }
            out.is_sorted = v;
            break;
        }
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

template <typename Reader>
static bool parse_data_page_header_v2(Reader &r, DataPageHeaderV2 &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 1:
            if (ctype != CT_I32 || !read_i32(r, out.num_values))
            {
                return false;
            }
            break;
        case 2:
            if (ctype != CT_I32 || !read_i32(r, out.num_nulls))
            {
                return false;
            }
            break;
        case 3:
            if (ctype != CT_I32 || !read_i32(r, out.num_rows))
            {
                return false;
            }
            break;
        case 4:
            if (ctype != CT_I32 || !read_i32(r, out.encoding))
            {
                return false;
            }
            break;
        case 5:
            if (ctype != CT_I32 || !read_i32(r, out.definition_levels_byte_length))
            {
                return false;
            }
            break;
        case 6:
            if (ctype != CT_I32 || !read_i32(r, out.repetition_levels_byte_length))
            {
                return false;
            }
            break;
        case 7: {
            bool v = false;
            if (!read_bool(r, bool_inline, bool_value, v))
            {
                return false;
            }
            out.is_compressed = v;
            break;
        }
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}

static bool parse_page_header(StreamReader &r, PageHeader &out)
{
    int16_t last = 0;
    while (true)
    {
        int16_t field_id = 0;
        uint8_t ctype = CT_STOP;
        bool bool_inline = false;
        bool bool_value = false;
        if (!read_field_header(r, last, field_id, ctype, bool_inline, bool_value))
        {
            return false;
        }
        if (ctype == CT_STOP)
        {
            return true;
        }
        switch (field_id)
        {
        case 1:
            if (ctype != CT_I32 || !read_i32(r, out.type))
            {
                return false;
            }
            break;
        case 2:
            if (ctype != CT_I32 || !read_i32(r, out.uncompressed_page_size))
            {
                return false;
            }
            break;
        case 3:
            if (ctype != CT_I32 || !read_i32(r, out.compressed_page_size))
            {
                return false;
            }
            break;
        case 5:
            if (ctype != CT_STRUCT || !parse_data_page_header(r, out.data_page_header))
            {
                return false;
            }
            out.has_data_page_header = true;
            break;
        case 7:
            if (ctype != CT_STRUCT || !parse_dictionary_page_header(r, out.dictionary_page_header))
            {
                return false;
            }
            out.has_dictionary_page_header = true;
            break;
        case 8:
            if (ctype != CT_STRUCT || !parse_data_page_header_v2(r, out.data_page_header_v2))
            {
                return false;
            }
            out.has_data_page_header_v2 = true;
            break;
        default:
            if (!skip_value(r, ctype, bool_inline))
            {
                return false;
            }
            break;
        }
    }
}
} // namespace pq
} // namespace

namespace
{
namespace pq
{
struct LeafInfo
{
    std::vector<std::string> path;
    int32_t type = -1;
    int16_t max_definition_level = 0;
    int16_t max_repetition_level = 0;
};

static bool decode_levels_v1(const uint8_t *data, std::size_t size, std::size_t &pos, uint32_t max_level,
                             int32_t level_encoding, std::size_t value_count, std::vector<uint32_t> &levels,
                             std::string &err);
static bool decode_levels_v2(const uint8_t *data, std::size_t size, uint32_t max_level, std::size_t value_count,
                             std::vector<uint32_t> &levels, std::string &err);
static bool decode_plain_byte_array(const uint8_t *data, std::size_t size, std::size_t count,
                                    std::vector<std::string> &out, std::string &err);
static bool decompress_page_payload(int32_t codec, const std::vector<uint8_t> &compressed, std::size_t expected_size,
                                    std::vector<uint8_t> &out, std::string &err);
static bool parse_file_metadata_from_footer(std::ifstream &in, uint64_t &file_size, FileMetaData &meta, std::string &err);
static bool build_leaf_infos(const std::vector<SchemaElement> &schema, std::vector<LeafInfo> &leaves);
static int choose_leaf_index(const std::vector<LeafInfo> &leaves, const std::string &text_field);
static int64_t resolve_column_start(const ColumnMetaData &col);
static std::string format_column_offsets(const ColumnMetaData &col);
static int64_t pick_next_column_start(const ColumnMetaData &col, int64_t failed_start);
static bool emit_text_values(int32_t encoding, const uint8_t *value_data, std::size_t value_size,
                             std::size_t page_value_count, const std::vector<uint32_t> &def_levels, int16_t max_def,
                             const std::vector<std::string> &dictionary,
                             const std::function<void(const std::string &)> &cb, std::string &err);
} // namespace pq
} // namespace

static bool read_parquet_records(const std::string &path, const std::string &text_field,
                                 const std::function<void(const std::string &)> &cb, std::string &err)
{
    using namespace pq;

    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        err = "failed to open parquet file: " + path;
        return false;
    }

    FileMetaData meta;
    uint64_t file_size = 0;
    if (!parse_file_metadata_from_footer(in, file_size, meta, err))
    {
        err += ": " + path;
        return false;
    }

    std::vector<LeafInfo> leaves;
    if (!build_leaf_infos(meta.schema, leaves))
    {
        err = "failed to build parquet schema leaf columns: " + path;
        return false;
    }
    if (leaves.empty())
    {
        return true;
    }

    int leaf_index = choose_leaf_index(leaves, text_field);
    if (leaf_index < 0 || static_cast<std::size_t>(leaf_index) >= leaves.size())
    {
        err = "parquet text field not found: " + text_field + " in " + path;
        return false;
    }
    const LeafInfo &leaf = leaves[static_cast<std::size_t>(leaf_index)];
    if (leaf.max_repetition_level > 0)
    {
        err = "repeated parquet field is not supported for text extraction: " + text_field + " in " + path;
        return false;
    }
    if (leaf.type != TYPE_BYTE_ARRAY)
    {
        err = "parquet field type is not BYTE_ARRAY for text extraction: " + text_field + " in " + path;
        return false;
    }

    for (const auto &rg : meta.row_groups)
    {
        if (leaf_index >= static_cast<int>(rg.columns.size()))
        {
            continue;
        }
        const ColumnMetaData &col = rg.columns[static_cast<std::size_t>(leaf_index)];
        if (col.num_values <= 0)
        {
            continue;
        }
        if (col.type != TYPE_BYTE_ARRAY)
        {
            err = "parquet column type mismatch for field: " + text_field + " in " + path;
            return false;
        }

        std::vector<int64_t> start_candidates;
        auto push_start = [&](int64_t v) {
            if (v < 0 || static_cast<uint64_t>(v) >= file_size)
            {
                return;
            }
            if (std::find(start_candidates.begin(), start_candidates.end(), v) == start_candidates.end())
            {
                start_candidates.push_back(v);
            }
        };
        int64_t primary_start = resolve_column_start(col);
        push_start(primary_start);
        push_start(pick_next_column_start(col, primary_start));
        if (start_candidates.empty())
        {
            err = "invalid parquet column start offset for field: " + text_field + " in " + path + " (" +
                  format_column_offsets(col) + ")";
            return false;
        }

        bool parsed_column = false;
        for (std::size_t start_idx = 0; start_idx < start_candidates.size() && !parsed_column; ++start_idx)
        {
            int64_t start = start_candidates[start_idx];

            in.clear();
            in.seekg(static_cast<std::streamoff>(start), std::ios::beg);
            if (!in)
            {
                if (start_idx + 1 < start_candidates.size())
                {
                    continue;
                }
                err = "failed to seek parquet column offset: " + path + " (start=" + std::to_string(start) + ")";
                return false;
            }
            StreamReader sr(in, static_cast<uint64_t>(start), file_size);

            std::vector<std::string> dictionary;
            int64_t values_seen = 0;
            bool retry_with_next_start = false;
            while (values_seen < col.num_values)
            {
                uint64_t page_header_offset = sr.pos();
                PageHeader page;
                if (!parse_page_header(sr, page))
                {
                    if (values_seen == 0 && start_idx + 1 < start_candidates.size())
                    {
                        retry_with_next_start = true;
                        break;
                    }
                    err = "failed to parse parquet page header at offset " + std::to_string(page_header_offset) +
                          " (start=" + std::to_string(start) + ", " + format_column_offsets(col) + ") in " + path;
                    return false;
                }
            if (page.compressed_page_size < 0 || page.uncompressed_page_size < 0)
            {
                err = "invalid parquet page size";
                return false;
            }

            std::vector<uint8_t> compressed(static_cast<std::size_t>(page.compressed_page_size));
            if (!compressed.empty() && !sr.read_exact(compressed.data(), compressed.size()))
            {
                err = "failed to read parquet page payload";
                return false;
            }

            if (page.type == PAGE_DICTIONARY)
            {
                if (!page.has_dictionary_page_header)
                {
                    err = "missing parquet dictionary page header";
                    return false;
                }
                if (page.dictionary_page_header.encoding != ENC_PLAIN)
                {
                    err = "unsupported parquet dictionary encoding";
                    return false;
                }
                std::vector<uint8_t> payload;
                if (!decompress_page_payload(col.codec, compressed, static_cast<std::size_t>(page.uncompressed_page_size),
                                             payload, err))
                {
                    return false;
                }
                if (!decode_plain_byte_array(payload.data(), payload.size(),
                                             static_cast<std::size_t>(page.dictionary_page_header.num_values), dictionary,
                                             err))
                {
                    return false;
                }
                continue;
            }

            if (page.type == PAGE_DATA)
            {
                if (!page.has_data_page_header)
                {
                    err = "missing parquet data page header";
                    return false;
                }
                if (page.data_page_header.num_values < 0)
                {
                    err = "invalid parquet data page value count";
                    return false;
                }
                std::size_t page_value_count = static_cast<std::size_t>(page.data_page_header.num_values);
                std::vector<uint8_t> payload;
                if (!decompress_page_payload(col.codec, compressed, static_cast<std::size_t>(page.uncompressed_page_size),
                                             payload, err))
                {
                    return false;
                }

                std::size_t pos = 0;
                std::vector<uint32_t> rep_levels;
                std::vector<uint32_t> def_levels;
                if (!decode_levels_v1(payload.data(), payload.size(), pos, static_cast<uint32_t>(leaf.max_repetition_level),
                                      page.data_page_header.repetition_level_encoding, page_value_count, rep_levels, err))
                {
                    return false;
                }
                if (!decode_levels_v1(payload.data(), payload.size(), pos, static_cast<uint32_t>(leaf.max_definition_level),
                                      page.data_page_header.definition_level_encoding, page_value_count, def_levels, err))
                {
                    return false;
                }

                if (leaf.max_repetition_level > 0)
                {
                    for (uint32_t r : rep_levels)
                    {
                        if (r != 0)
                        {
                            err = "repeated parquet pages are not supported";
                            return false;
                        }
                    }
                }

                if (pos > payload.size())
                {
                    err = "invalid parquet data page payload";
                    return false;
                }
                if (!emit_text_values(page.data_page_header.encoding, payload.data() + pos, payload.size() - pos,
                                      page_value_count, def_levels, leaf.max_definition_level, dictionary, cb, err))
                {
                    return false;
                }
                values_seen += static_cast<int64_t>(page_value_count);
                continue;
            }

            if (page.type == PAGE_DATA_V2)
            {
                if (!page.has_data_page_header_v2)
                {
                    err = "missing parquet data page v2 header";
                    return false;
                }
                if (page.data_page_header_v2.num_values < 0)
                {
                    err = "invalid parquet data page v2 value count";
                    return false;
                }
                std::size_t page_value_count = static_cast<std::size_t>(page.data_page_header_v2.num_values);
                std::size_t rep_len =
                    static_cast<std::size_t>(std::max(0, page.data_page_header_v2.repetition_levels_byte_length));
                std::size_t def_len =
                    static_cast<std::size_t>(std::max(0, page.data_page_header_v2.definition_levels_byte_length));
                if (rep_len + def_len > compressed.size())
                {
                    err = "invalid parquet data page v2 level lengths";
                    return false;
                }

                const uint8_t *rep_ptr = compressed.data();
                const uint8_t *def_ptr = compressed.data() + rep_len;
                const uint8_t *val_ptr = compressed.data() + rep_len + def_len;
                std::size_t val_comp_size = compressed.size() - rep_len - def_len;

                std::vector<uint32_t> rep_levels;
                std::vector<uint32_t> def_levels;
                if (!decode_levels_v2(rep_ptr, rep_len, static_cast<uint32_t>(leaf.max_repetition_level), page_value_count,
                                      rep_levels, err))
                {
                    return false;
                }
                if (!decode_levels_v2(def_ptr, def_len, static_cast<uint32_t>(leaf.max_definition_level), page_value_count,
                                      def_levels, err))
                {
                    return false;
                }

                if (leaf.max_repetition_level > 0)
                {
                    for (uint32_t r : rep_levels)
                    {
                        if (r != 0)
                        {
                            err = "repeated parquet pages are not supported";
                            return false;
                        }
                    }
                }

                std::vector<uint8_t> value_payload;
                const uint8_t *value_data = val_ptr;
                std::size_t value_size = val_comp_size;
                if (page.data_page_header_v2.is_compressed)
                {
                    int32_t expected = page.uncompressed_page_size -
                                       page.data_page_header_v2.repetition_levels_byte_length -
                                       page.data_page_header_v2.definition_levels_byte_length;
                    if (expected < 0)
                    {
                        err = "invalid parquet data page v2 uncompressed size";
                        return false;
                    }
                    std::vector<uint8_t> compressed_values(val_ptr, val_ptr + val_comp_size);
                    if (!decompress_page_payload(col.codec, compressed_values, static_cast<std::size_t>(expected),
                                                 value_payload, err))
                    {
                        return false;
                    }
                    value_data = value_payload.data();
                    value_size = value_payload.size();
                }

                if (!emit_text_values(page.data_page_header_v2.encoding, value_data, value_size, page_value_count,
                                      def_levels, leaf.max_definition_level, dictionary, cb, err))
                {
                    return false;
                }
                values_seen += static_cast<int64_t>(page_value_count);
                continue;
            }

            err = "unsupported parquet page type: " + std::to_string(page.type);
            return false;
            }

            if (retry_with_next_start)
            {
                continue;
            }
            parsed_column = true;
        }

        if (!parsed_column)
        {
            err = "failed to parse parquet column pages for field: " + text_field + " in " + path + " (" +
                  format_column_offsets(col) + ")";
            return false;
        }
    }

    return true;
}

namespace
{
namespace pq
{
static uint32_t load_u32_le(const uint8_t *p)
{
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static unsigned bit_width(uint32_t v)
{
    unsigned w = 0;
    while (v > 0)
    {
        ++w;
        v >>= 1;
    }
    return w;
}

static bool read_varint_from_bytes(const uint8_t *data, std::size_t size, std::size_t &pos, uint64_t &out)
{
    out = 0;
    int shift = 0;
    for (int i = 0; i < 10; ++i)
    {
        if (pos >= size)
        {
            return false;
        }
        uint8_t b = data[pos++];
        out |= static_cast<uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0)
        {
            return true;
        }
        shift += 7;
    }
    return false;
}

static bool decode_rle_bitpacked(const uint8_t *data, std::size_t size, unsigned bw, std::size_t value_count,
                                 std::vector<uint32_t> &out, std::size_t &consumed, std::string &err)
{
    out.clear();
    out.reserve(value_count);
    consumed = 0;
    std::size_t pos = 0;

    if (bw == 0)
    {
        out.assign(value_count, 0);
        consumed = 0;
        return true;
    }
    if (bw > 32)
    {
        err = "unsupported bit width in rle/bitpacked stream";
        return false;
    }

    while (out.size() < value_count)
    {
        uint64_t header = 0;
        if (!read_varint_from_bytes(data, size, pos, header))
        {
            err = "invalid rle/bitpacked header";
            return false;
        }
        if ((header & 1U) == 0)
        {
            uint64_t run_len = header >> 1U;
            std::size_t byte_width = (bw + 7U) / 8U;
            if (pos + byte_width > size)
            {
                err = "truncated rle run";
                return false;
            }
            uint32_t value = 0;
            for (std::size_t i = 0; i < byte_width; ++i)
            {
                value |= static_cast<uint32_t>(data[pos + i]) << (8U * static_cast<unsigned>(i));
            }
            pos += byte_width;
            std::size_t take = static_cast<std::size_t>(std::min<uint64_t>(run_len, value_count - out.size()));
            out.insert(out.end(), take, value);
        }
        else
        {
            uint64_t groups = header >> 1U;
            uint64_t run_len = groups * 8ULL;
            uint64_t total_bits = run_len * static_cast<uint64_t>(bw);
            std::size_t byte_len = static_cast<std::size_t>((total_bits + 7ULL) / 8ULL);
            if (pos + byte_len > size)
            {
                err = "truncated bit-packed run";
                return false;
            }
            std::size_t take = static_cast<std::size_t>(std::min<uint64_t>(run_len, value_count - out.size()));
            for (std::size_t i = 0; i < take; ++i)
            {
                uint32_t value = 0;
                uint64_t base = static_cast<uint64_t>(i) * static_cast<uint64_t>(bw);
                for (unsigned b = 0; b < bw; ++b)
                {
                    uint64_t bit_index = base + b;
                    std::size_t byte_index = static_cast<std::size_t>(bit_index / 8ULL);
                    unsigned bit_offset = static_cast<unsigned>(bit_index % 8ULL);
                    if (((data[pos + byte_index] >> bit_offset) & 0x01U) != 0U)
                    {
                        value |= (1U << b);
                    }
                }
                out.push_back(value);
            }
            pos += byte_len;
        }
    }

    consumed = pos;
    return true;
}

static bool decode_levels_v1(const uint8_t *data, std::size_t size, std::size_t &pos, uint32_t max_level,
                             int32_t level_encoding, std::size_t value_count, std::vector<uint32_t> &levels,
                             std::string &err)
{
    levels.clear();
    if (max_level == 0)
    {
        levels.assign(value_count, 0);
        return true;
    }
    if (level_encoding != ENC_RLE)
    {
        err = "unsupported parquet level encoding in data page v1";
        return false;
    }
    if (pos + 4 > size)
    {
        err = "truncated parquet level length";
        return false;
    }
    uint32_t len = load_u32_le(data + pos);
    pos += 4;
    if (pos + len > size)
    {
        err = "truncated parquet level payload";
        return false;
    }
    std::size_t consumed = 0;
    if (!decode_rle_bitpacked(data + pos, len, bit_width(max_level), value_count, levels, consumed, err))
    {
        return false;
    }
    pos += len;
    return true;
}

static bool decode_levels_v2(const uint8_t *data, std::size_t size, uint32_t max_level, std::size_t value_count,
                             std::vector<uint32_t> &levels, std::string &err)
{
    levels.clear();
    if (max_level == 0)
    {
        levels.assign(value_count, 0);
        return true;
    }
    std::size_t consumed = 0;
    if (!decode_rle_bitpacked(data, size, bit_width(max_level), value_count, levels, consumed, err))
    {
        return false;
    }
    return true;
}

static bool decode_plain_byte_array(const uint8_t *data, std::size_t size, std::size_t count,
                                    std::vector<std::string> &out, std::string &err)
{
    out.clear();
    out.reserve(count);
    std::size_t pos = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (pos + 4 > size)
        {
            err = "truncated parquet plain byte array length";
            return false;
        }
        uint32_t len = load_u32_le(data + pos);
        pos += 4;
        if (pos + len > size)
        {
            err = "truncated parquet plain byte array payload";
            return false;
        }
        out.emplace_back(reinterpret_cast<const char *>(data + pos), static_cast<std::size_t>(len));
        pos += len;
    }
    return true;
}

static bool decode_dictionary_ids(const uint8_t *data, std::size_t size, std::size_t count, std::vector<uint32_t> &ids,
                                  std::string &err)
{
    ids.clear();
    if (count == 0)
    {
        return true;
    }
    if (size < 1)
    {
        err = "missing parquet dictionary id bit width";
        return false;
    }
    unsigned bw = data[0];
    std::size_t consumed = 0;
    if (decode_rle_bitpacked(data + 1, size - 1, bw, count, ids, consumed, err))
    {
        return true;
    }

    if (size >= 5)
    {
        uint32_t len = load_u32_le(data + 1);
        if (static_cast<std::size_t>(len) <= size - 5)
        {
            std::string fallback_err;
            if (decode_rle_bitpacked(data + 5, static_cast<std::size_t>(len), bw, count, ids, consumed, fallback_err))
            {
                return true;
            }
            err = fallback_err;
            return false;
        }
    }
    return false;
}

static bool gzip_decompress(const uint8_t *src, std::size_t src_size, std::size_t expected_size,
                            std::vector<uint8_t> &out, std::string &err)
{
    z_stream strm;
    std::memset(&strm, 0, sizeof(strm));
    strm.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(src));
    strm.avail_in = static_cast<uInt>(src_size);
    if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK)
    {
        err = "gzip inflateInit2 failed";
        return false;
    }

    std::size_t cap = expected_size > 0 ? expected_size : std::max<std::size_t>(src_size * 3U, 1024U);
    out.assign(cap, 0);
    bool ok = true;

    while (true)
    {
        if (strm.total_out == out.size())
        {
            if (out.size() > (1ULL << 30))
            {
                ok = false;
                err = "gzip output is too large";
                break;
            }
            out.resize(out.size() * 2U);
        }
        strm.next_out = reinterpret_cast<Bytef *>(out.data() + strm.total_out);
        strm.avail_out = static_cast<uInt>(out.size() - strm.total_out);
        int ret = inflate(&strm, Z_NO_FLUSH);
        if (ret == Z_STREAM_END)
        {
            break;
        }
        if (ret != Z_OK)
        {
            ok = false;
            err = "gzip inflate failed";
            break;
        }
    }

    out.resize(strm.total_out);
    inflateEnd(&strm);
    return ok;
}

static bool snappy_decompress(const uint8_t *src, std::size_t src_size, std::vector<uint8_t> &out, std::string &err)
{
    out.clear();
    std::size_t pos = 0;
    uint64_t decoded_len = 0;
    if (!read_varint_from_bytes(src, src_size, pos, decoded_len))
    {
        err = "invalid snappy varint length";
        return false;
    }
    if (decoded_len > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        err = "snappy output length overflow";
        return false;
    }
    out.reserve(static_cast<std::size_t>(decoded_len));

    while (pos < src_size)
    {
        uint8_t tag = src[pos++];
        uint8_t tag_type = static_cast<uint8_t>(tag & 0x03);
        if (tag_type == 0)
        {
            uint32_t lit_len = static_cast<uint32_t>(tag >> 2);
            if (lit_len < 60)
            {
                lit_len += 1;
            }
            else
            {
                uint32_t n = lit_len - 59;
                if (n < 1 || n > 4 || pos + n > src_size)
                {
                    err = "invalid snappy literal length";
                    return false;
                }
                lit_len = 0;
                for (uint32_t i = 0; i < n; ++i)
                {
                    lit_len |= static_cast<uint32_t>(src[pos + i]) << (8U * i);
                }
                pos += n;
                lit_len += 1;
            }
            if (pos + lit_len > src_size)
            {
                err = "truncated snappy literal";
                return false;
            }
            out.insert(out.end(), src + pos, src + pos + lit_len);
            pos += lit_len;
            continue;
        }

        uint32_t len = 0;
        uint32_t offset = 0;
        if (tag_type == 1)
        {
            if (pos >= src_size)
            {
                err = "truncated snappy copy(1)";
                return false;
            }
            len = 4U + ((tag >> 2U) & 0x07U);
            offset = (static_cast<uint32_t>(tag & 0xE0U) << 3U) | static_cast<uint32_t>(src[pos++]);
        }
        else if (tag_type == 2)
        {
            if (pos + 2 > src_size)
            {
                err = "truncated snappy copy(2)";
                return false;
            }
            len = 1U + (tag >> 2U);
            offset = static_cast<uint32_t>(src[pos]) | (static_cast<uint32_t>(src[pos + 1]) << 8U);
            pos += 2;
        }
        else
        {
            if (pos + 4 > src_size)
            {
                err = "truncated snappy copy(4)";
                return false;
            }
            len = 1U + (tag >> 2U);
            offset = static_cast<uint32_t>(src[pos]) | (static_cast<uint32_t>(src[pos + 1]) << 8U) |
                     (static_cast<uint32_t>(src[pos + 2]) << 16U) | (static_cast<uint32_t>(src[pos + 3]) << 24U);
            pos += 4;
        }

        if (offset == 0 || offset > out.size())
        {
            err = "invalid snappy copy offset";
            return false;
        }
        std::size_t start = out.size() - offset;
        for (uint32_t i = 0; i < len; ++i)
        {
            out.push_back(out[start + i]);
        }
    }

    if (out.size() != static_cast<std::size_t>(decoded_len))
    {
        err = "snappy output length mismatch";
        return false;
    }
    return true;
}

static bool decompress_page_payload(int32_t codec, const std::vector<uint8_t> &compressed, std::size_t expected_size,
                                    std::vector<uint8_t> &out, std::string &err)
{
    if (codec == CODEC_UNCOMPRESSED)
    {
        out = compressed;
        return true;
    }
    if (codec == CODEC_GZIP)
    {
        return gzip_decompress(compressed.data(), compressed.size(), expected_size, out, err);
    }
    if (codec == CODEC_SNAPPY)
    {
        return snappy_decompress(compressed.data(), compressed.size(), out, err);
    }
    err = "unsupported parquet codec: " + std::to_string(codec);
    return false;
}

static bool parse_file_metadata_from_footer(std::ifstream &in, uint64_t &file_size, FileMetaData &meta, std::string &err)
{
    in.clear();
    in.seekg(0, std::ios::end);
    std::streamoff end = in.tellg();
    if (end < 12)
    {
        err = "invalid parquet file: too small";
        return false;
    }
    file_size = static_cast<uint64_t>(end);

    in.seekg(static_cast<std::streamoff>(file_size - 8), std::ios::beg);
    uint8_t tail[8] = {0};
    in.read(reinterpret_cast<char *>(tail), 8);
    if (!in)
    {
        err = "failed to read parquet footer";
        return false;
    }
    if (std::memcmp(tail + 4, "PAR1", 4) != 0)
    {
        err = "invalid parquet magic in footer";
        return false;
    }
    uint32_t meta_len = load_u32_le(tail);
    if (static_cast<uint64_t>(meta_len) + 8ULL > file_size)
    {
        err = "invalid parquet metadata length";
        return false;
    }

    uint64_t meta_start = file_size - 8ULL - static_cast<uint64_t>(meta_len);
    in.seekg(static_cast<std::streamoff>(meta_start), std::ios::beg);
    std::vector<uint8_t> meta_buf(meta_len);
    if (meta_len > 0)
    {
        in.read(reinterpret_cast<char *>(meta_buf.data()), static_cast<std::streamsize>(meta_len));
        if (!in)
        {
            err = "failed to read parquet metadata";
            return false;
        }
    }

    BufferReader br(meta_buf.data(), meta_buf.size());
    if (!parse_file_metadata(br, meta))
    {
        err = "failed to parse parquet metadata";
        return false;
    }
    return true;
}

static bool build_leaf_infos_recursive(const std::vector<SchemaElement> &schema, std::size_t idx,
                                       std::vector<std::string> &path, int16_t parent_def, int16_t parent_rep,
                                       std::vector<LeafInfo> &leaves, std::size_t &next_idx)
{
    if (idx >= schema.size())
    {
        return false;
    }
    const SchemaElement &node = schema[idx];
    int16_t def = parent_def;
    int16_t rep = parent_rep;

    if (idx != 0)
    {
        if (node.repetition_type == REP_OPTIONAL)
        {
            ++def;
        }
        else if (node.repetition_type == REP_REPEATED)
        {
            ++def;
            ++rep;
        }
        path.push_back(node.name);
    }

    std::size_t cursor = idx + 1;
    if (node.num_children <= 0)
    {
        LeafInfo leaf;
        leaf.path = path;
        leaf.type = node.type;
        leaf.max_definition_level = def;
        leaf.max_repetition_level = rep;
        leaves.push_back(std::move(leaf));
    }
    else
    {
        for (int32_t i = 0; i < node.num_children; ++i)
        {
            if (!build_leaf_infos_recursive(schema, cursor, path, def, rep, leaves, cursor))
            {
                return false;
            }
        }
    }

    if (idx != 0)
    {
        path.pop_back();
    }
    next_idx = cursor;
    return true;
}

static bool build_leaf_infos(const std::vector<SchemaElement> &schema, std::vector<LeafInfo> &leaves)
{
    leaves.clear();
    if (schema.empty())
    {
        return true;
    }
    std::vector<std::string> path;
    std::size_t next = 0;
    return build_leaf_infos_recursive(schema, 0, path, 0, 0, leaves, next);
}

static std::vector<std::string> split_dot_path(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == '.')
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

static int choose_leaf_index(const std::vector<LeafInfo> &leaves, const std::string &text_field)
{
    auto parts = split_dot_path(text_field);
    if (parts.size() > 1)
    {
        for (std::size_t i = 0; i < leaves.size(); ++i)
        {
            if (leaves[i].path == parts)
            {
                return static_cast<int>(i);
            }
        }
    }
    int direct = -1;
    int fallback = -1;
    for (std::size_t i = 0; i < leaves.size(); ++i)
    {
        if (leaves[i].path.empty())
        {
            continue;
        }
        if (leaves[i].path.back() != text_field)
        {
            continue;
        }
        if (leaves[i].path.size() == 1)
        {
            direct = static_cast<int>(i);
            break;
        }
        if (fallback < 0)
        {
            fallback = static_cast<int>(i);
        }
    }
    return direct >= 0 ? direct : fallback;
}

static int64_t resolve_column_start(const ColumnMetaData &col)
{
    int64_t start = std::numeric_limits<int64_t>::max();
    auto consider_page = [&](int64_t v) {
        if (v >= 0 && v < start)
        {
            start = v;
        }
    };
    consider_page(col.dictionary_page_offset);
    consider_page(col.data_page_offset);
    if (start != std::numeric_limits<int64_t>::max())
    {
        return start;
    }
    if (col.file_offset >= 0)
    {
        return col.file_offset;
    }
    return -1;
}

static std::string format_column_offsets(const ColumnMetaData &col)
{
    return "data=" + std::to_string(col.data_page_offset) + ", dict=" + std::to_string(col.dictionary_page_offset) +
           ", file=" + std::to_string(col.file_offset);
}

static int64_t pick_next_column_start(const ColumnMetaData &col, int64_t failed_start)
{
    std::vector<int64_t> cands;
    auto push = [&](int64_t v) {
        if (v < 0 || v == failed_start)
        {
            return;
        }
        if (std::find(cands.begin(), cands.end(), v) == cands.end())
        {
            cands.push_back(v);
        }
    };
    if (col.dictionary_page_offset >= 0 && col.data_page_offset >= 0)
    {
        if (col.dictionary_page_offset <= col.data_page_offset)
        {
            push(col.dictionary_page_offset);
            push(col.data_page_offset);
        }
        else
        {
            push(col.data_page_offset);
            push(col.dictionary_page_offset);
        }
    }
    else
    {
        push(col.dictionary_page_offset);
        push(col.data_page_offset);
    }
    push(col.file_offset);
    if (cands.empty())
    {
        return -1;
    }
    return cands.front();
}

static bool emit_text_values(int32_t encoding, const uint8_t *value_data, std::size_t value_size,
                             std::size_t page_value_count, const std::vector<uint32_t> &def_levels, int16_t max_def,
                             const std::vector<std::string> &dictionary,
                             const std::function<void(const std::string &)> &cb, std::string &err)
{
    if (max_def > 0 && def_levels.size() != page_value_count)
    {
        err = "definition level size mismatch";
        return false;
    }

    std::size_t non_null_count = 0;
    if (max_def == 0)
    {
        non_null_count = page_value_count;
    }
    else
    {
        for (auto d : def_levels)
        {
            if (d == static_cast<uint32_t>(max_def))
            {
                ++non_null_count;
            }
        }
    }

    if (encoding == ENC_PLAIN)
    {
        std::vector<std::string> values;
        if (!decode_plain_byte_array(value_data, value_size, non_null_count, values, err))
        {
            return false;
        }
        std::size_t vidx = 0;
        for (std::size_t i = 0; i < page_value_count; ++i)
        {
            bool present = (max_def == 0) || (def_levels[i] == static_cast<uint32_t>(max_def));
            if (!present)
            {
                continue;
            }
            if (vidx >= values.size())
            {
                err = "plain value count mismatch";
                return false;
            }
            if (!values[vidx].empty())
            {
                cb(values[vidx]);
            }
            ++vidx;
        }
        return true;
    }

    if (encoding == ENC_RLE_DICTIONARY || encoding == ENC_PLAIN_DICTIONARY)
    {
        if (dictionary.empty())
        {
            err = "dictionary encoded page without dictionary";
            return false;
        }
        std::vector<uint32_t> ids;
        if (!decode_dictionary_ids(value_data, value_size, non_null_count, ids, err))
        {
            return false;
        }
        std::size_t iidx = 0;
        for (std::size_t i = 0; i < page_value_count; ++i)
        {
            bool present = (max_def == 0) || (def_levels[i] == static_cast<uint32_t>(max_def));
            if (!present)
            {
                continue;
            }
            if (iidx >= ids.size())
            {
                err = "dictionary id count mismatch";
                return false;
            }
            uint32_t dict_id = ids[iidx++];
            if (dict_id >= dictionary.size())
            {
                err = "dictionary id out of range";
                return false;
            }
            if (!dictionary[dict_id].empty())
            {
                cb(dictionary[dict_id]);
            }
        }
        return true;
    }

    err = "unsupported parquet value encoding: " + std::to_string(encoding);
    return false;
}
} // namespace pq
} // namespace

