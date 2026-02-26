#include "train_backend_common.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <queue>

#include "tokenflux_lib.hpp"

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

std::vector<Word> build_words(const GlobalCountMap &global_counts, const std::unordered_map<uint32_t, int> &cp_to_id,
                              std::size_t min_freq)
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

struct HeapItem
{
    uint64_t key;
    uint64_t count;
};

struct HeapCmp
{
    bool operator()(const HeapItem &a, const HeapItem &b) const
    {
        return a.count < b.count;
    }
};

void train_bpe(std::vector<Word> &words, std::vector<std::string> &id_to_symbol, std::vector<std::string> &merges_out,
               std::size_t target_vocab, std::size_t min_pair_freq, std::size_t max_pair_entries)
{
    std::unordered_map<uint64_t, uint64_t> pair_counts;
    std::size_t reserve_hint = words.size() * 2 + 1024;
    if (max_pair_entries > 0)
    {
        reserve_hint = std::min<std::size_t>(reserve_hint, max_pair_entries);
    }
    pair_counts.reserve(reserve_hint);
    std::unordered_map<uint64_t, std::vector<uint32_t>> pair_words;
    pair_words.reserve(reserve_hint);
    std::size_t skipped_pairs_due_to_cap = 0;

    auto can_track_pair = [&](uint64_t key) -> bool {
        if (max_pair_entries == 0)
        {
            return true;
        }
        if (pair_counts.find(key) != pair_counts.end())
        {
            return true;
        }
        return pair_counts.size() < max_pair_entries;
    };

    for (uint32_t idx = 0; idx < words.size(); ++idx)
    {
        const auto &syms = words[idx].symbols;
        if (syms.size() < 2)
        {
            continue;
        }
        std::vector<uint64_t> keys;
        keys.reserve(syms.size() - 1);
        for (std::size_t i = 0; i + 1 < syms.size(); ++i)
        {
            uint64_t key = pair_key(syms[i], syms[i + 1]);
            if (!can_track_pair(key))
            {
                ++skipped_pairs_due_to_cap;
                continue;
            }
            pair_counts[key] += words[idx].freq;
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        for (auto key : keys)
        {
            pair_words[key].push_back(idx);
        }
    }
    if (skipped_pairs_due_to_cap > 0)
    {
        std::cerr << "pair cap active: skipped " << skipped_pairs_due_to_cap
                  << " new pair keys while building initial stats\n";
    }

    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp> heap;
    for (const auto &kv : pair_counts)
    {
        heap.push({kv.first, kv.second});
    }

    const std::size_t initial_vocab = id_to_symbol.size();
    std::size_t max_merges = 0;
    if (target_vocab > initial_vocab)
    {
        max_merges = target_vocab - initial_vocab;
    }

    std::size_t merges_done = 0;
    while (merges_done < max_merges && !heap.empty())
    {
        HeapItem top = heap.top();
        heap.pop();
        auto it = pair_counts.find(top.key);
        if (it == pair_counts.end())
        {
            continue;
        }
        if (it->second != top.count)
        {
            continue;
        }
        if (top.count < min_pair_freq)
        {
            break;
        }
        int a = static_cast<int>(top.key >> 32);
        int b = static_cast<int>(top.key & 0xFFFFFFFFu);

        std::string merged_sym = id_to_symbol[a] + id_to_symbol[b];
        int new_id = static_cast<int>(id_to_symbol.size());
        id_to_symbol.push_back(std::move(merged_sym));
        merges_out.push_back(id_to_symbol[a] + " " + id_to_symbol[b]);

        auto words_it = pair_words.find(top.key);
        if (words_it != pair_words.end())
        {
            const auto &wlist = words_it->second;
            for (uint32_t widx : wlist)
            {
                auto &w = words[widx];
                if (w.symbols.size() < 2)
                {
                    continue;
                }
                std::vector<int> new_syms;
                new_syms.reserve(w.symbols.size());
                bool changed = false;
                for (std::size_t i = 0; i < w.symbols.size(); ++i)
                {
                    if (i + 1 < w.symbols.size() && w.symbols[i] == a && w.symbols[i + 1] == b)
                    {
                        int prev = new_syms.empty() ? -1 : new_syms.back();
                        int next = (i + 2 < w.symbols.size()) ? w.symbols[i + 2] : -1;

                        if (prev != -1)
                        {
                            uint64_t k_prev = pair_key(prev, a);
                            auto it_prev = pair_counts.find(k_prev);
                            if (it_prev != pair_counts.end())
                            {
                                auto &cnt = it_prev->second;
                                cnt = cnt >= w.freq ? cnt - w.freq : 0;
                                heap.push({k_prev, cnt});
                            }
                        }
                        if (next != -1)
                        {
                            uint64_t k_next = pair_key(b, next);
                            auto it_next = pair_counts.find(k_next);
                            if (it_next != pair_counts.end())
                            {
                                auto &cnt = it_next->second;
                                cnt = cnt >= w.freq ? cnt - w.freq : 0;
                                heap.push({k_next, cnt});
                            }
                        }

                        if (prev != -1)
                        {
                            uint64_t k_new_prev = pair_key(prev, new_id);
                            auto it_new_prev = pair_counts.find(k_new_prev);
                            if (it_new_prev != pair_counts.end())
                            {
                                auto &cnt = it_new_prev->second;
                                cnt += w.freq;
                                heap.push({k_new_prev, cnt});
                                auto &vec = pair_words[k_new_prev];
                                if (vec.empty() || vec.back() != widx)
                                {
                                    vec.push_back(widx);
                                }
                            }
                            else if (can_track_pair(k_new_prev))
                            {
                                pair_counts.emplace(k_new_prev, w.freq);
                                heap.push({k_new_prev, w.freq});
                                pair_words[k_new_prev].push_back(widx);
                            }
                            else
                            {
                                ++skipped_pairs_due_to_cap;
                            }
                        }
                        if (next != -1)
                        {
                            uint64_t k_new_next = pair_key(new_id, next);
                            auto it_new_next = pair_counts.find(k_new_next);
                            if (it_new_next != pair_counts.end())
                            {
                                auto &cnt = it_new_next->second;
                                cnt += w.freq;
                                heap.push({k_new_next, cnt});
                                auto &vec = pair_words[k_new_next];
                                if (vec.empty() || vec.back() != widx)
                                {
                                    vec.push_back(widx);
                                }
                            }
                            else if (can_track_pair(k_new_next))
                            {
                                pair_counts.emplace(k_new_next, w.freq);
                                heap.push({k_new_next, w.freq});
                                pair_words[k_new_next].push_back(widx);
                            }
                            else
                            {
                                ++skipped_pairs_due_to_cap;
                            }
                        }

                        new_syms.push_back(new_id);
                        ++i;
                        changed = true;
                    }
                    else
                    {
                        new_syms.push_back(w.symbols[i]);
                    }
                }
                if (changed)
                {
                    w.symbols.swap(new_syms);
                }
            }
        }
        pair_counts[top.key] = 0;
        ++merges_done;
        if (merges_done % 1000 == 0)
        {
            std::cerr << "Merges: " << merges_done << "/" << max_merges << "\n";
        }
    }
    if (skipped_pairs_due_to_cap > 0)
    {
        std::cerr << "pair cap total skipped new pair keys: " << skipped_pairs_due_to_cap << "\n";
    }
}
