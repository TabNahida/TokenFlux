#include "train_backend_common.h"

#include <algorithm>
#include <cstdio>

#include "tokenflux_lib.h"

bool starts_with(const std::string &s, const std::string &prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

void append_utf8(uint32_t cp, std::string &out)
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

static bool is_space(uint32_t cp)
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
        uint32_t cp = 0;
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
        if (is_space(cp))
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

std::vector<std::string> split_codepoints(const std::string &text)
{
    std::vector<std::string> cps;
    std::size_t i = 0;
    while (i < text.size())
    {
        uint32_t cp = 0;
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

std::vector<std::string> make_special_tokens(const Config &cfg)
{
    std::vector<std::string> specials = cfg.special_tokens;
    if (std::find(specials.begin(), specials.end(), cfg.unk_token) == specials.end())
    {
        specials.push_back(cfg.unk_token);
    }
    return specials;
}

std::size_t calc_pair_target_vocab(const Config &cfg, std::size_t base_symbols, std::size_t special_count)
{
    std::size_t target_vocab = cfg.vocab_size;
    if (special_count < target_vocab)
    {
        target_vocab -= special_count;
    }
    else
    {
        target_vocab = base_symbols;
    }
    if (target_vocab < base_symbols)
    {
        target_vocab = base_symbols;
    }
    return target_vocab;
}

void maybe_reduce_local(LocalCountMap &local_counts, const Config &cfg, std::size_t &reduce_counter,
                        std::size_t local_entry_cap)
{
    ++reduce_counter;
    if (cfg.chunk_docs > 0 && reduce_counter >= cfg.chunk_docs)
    {
        std::size_t keep = cfg.top_k;
        if (local_entry_cap > 0 && (keep == 0 || keep > local_entry_cap))
        {
            keep = local_entry_cap;
        }
        if (keep > 0)
        {
            local_counts = reduce_top_k(local_counts, keep);
        }
        reduce_counter = 0;
    }
    if (local_entry_cap > 0 && local_counts.size() > local_entry_cap)
    {
        local_counts = reduce_top_k(local_counts, local_entry_cap);
    }
}

int ensure_symbol(const std::string &token, SymbolMap &token_to_id, std::vector<std::string> &id_to_token)
{
    auto it = token_to_id.find(token);
    if (it != token_to_id.end())
    {
        return it->second;
    }
    int id = static_cast<int>(id_to_token.size());
    id_to_token.push_back(token);
    token_to_id.emplace(id_to_token.back(), id);
    return id;
}

std::vector<Word> build_words_from_tokens(const GlobalCountMap &global_counts,
                                          const std::unordered_map<uint32_t, int> &cp_to_id, std::size_t min_freq)
{
    std::vector<Word> words;
    words.reserve(global_counts.size());
    for (const auto &kv : global_counts)
    {
        if (kv.second < min_freq)
        {
            continue;
        }
        std::vector<int> symbols;
        std::size_t i = 0;
        while (i < kv.first.size())
        {
            uint32_t cp = 0;
            if (!next_codepoint(kv.first, i, cp))
            {
                break;
            }
            auto it = cp_to_id.find(cp);
            if (it == cp_to_id.end())
            {
                symbols.clear();
                break;
            }
            symbols.push_back(it->second);
        }
        if (!symbols.empty())
        {
            words.push_back({std::move(symbols), kv.second});
        }
    }
    return words;
}

void append_symbols_to_vocab(const std::vector<std::string> &specials, const std::vector<std::string> &symbols,
                             std::vector<std::string> &id_to_token)
{
    id_to_token = specials;
    id_to_token.reserve(specials.size() + symbols.size());
    for (const auto &sym : symbols)
    {
        if (std::find(id_to_token.begin(), id_to_token.end(), sym) == id_to_token.end())
        {
            id_to_token.push_back(sym);
        }
    }
}

uint64_t pair_key(int a, int b)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

