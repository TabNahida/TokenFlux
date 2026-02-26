#include "train_backend.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "train_backend_common.h"

namespace
{
struct WPWord
{
    std::vector<int> symbols;
    uint64_t freq = 0;
};

void train_wordpiece(std::vector<WPWord> &words, std::vector<std::string> &id_to_symbol, SymbolMap &symbol_to_id,
                     std::size_t target_vocab, std::size_t min_pair_freq, const std::string &prefix)
{
    std::size_t merges_done = 0;
    while (id_to_symbol.size() < target_vocab)
    {
        std::unordered_map<uint64_t, uint64_t> pair_counts;
        std::unordered_map<int, uint64_t> symbol_counts;
        pair_counts.reserve(words.size() * 2 + 1024);
        symbol_counts.reserve(id_to_symbol.size() * 2 + 1024);

        for (const auto &w : words)
        {
            if (w.symbols.empty())
            {
                continue;
            }
            for (int sid : w.symbols)
            {
                symbol_counts[sid] += w.freq;
            }
            for (std::size_t i = 0; i + 1 < w.symbols.size(); ++i)
            {
                pair_counts[pair_key(w.symbols[i], w.symbols[i + 1])] += w.freq;
            }
        }

        uint64_t best_key = 0;
        uint64_t best_pair_freq = 0;
        double best_score = -1.0;
        for (const auto &kv : pair_counts)
        {
            if (kv.second < min_pair_freq)
            {
                continue;
            }
            int a = static_cast<int>(kv.first >> 32);
            int b = static_cast<int>(kv.first & 0xFFFFFFFFu);
            uint64_t left = symbol_counts[a];
            uint64_t right = symbol_counts[b];
            if (left == 0 || right == 0)
            {
                continue;
            }
            double score = static_cast<double>(kv.second) / (static_cast<double>(left) * static_cast<double>(right));
            if (score > best_score || (score == best_score && kv.second > best_pair_freq))
            {
                best_score = score;
                best_pair_freq = kv.second;
                best_key = kv.first;
            }
        }

        if (best_score < 0.0)
        {
            break;
        }

        int a = static_cast<int>(best_key >> 32);
        int b = static_cast<int>(best_key & 0xFFFFFFFFu);
        std::string rhs = id_to_symbol[b];
        if (!prefix.empty() && starts_with(rhs, prefix))
        {
            rhs = rhs.substr(prefix.size());
        }
        std::string merged = id_to_symbol[a] + rhs;
        int new_id = ensure_symbol(merged, symbol_to_id, id_to_symbol);

        bool changed_any = false;
        for (auto &w : words)
        {
            if (w.symbols.size() < 2)
            {
                continue;
            }
            std::vector<int> merged_syms;
            merged_syms.reserve(w.symbols.size());
            bool changed = false;
            for (std::size_t i = 0; i < w.symbols.size();)
            {
                if (i + 1 < w.symbols.size() && w.symbols[i] == a && w.symbols[i + 1] == b)
                {
                    merged_syms.push_back(new_id);
                    i += 2;
                    changed = true;
                    continue;
                }
                merged_syms.push_back(w.symbols[i]);
                ++i;
            }
            if (changed)
            {
                changed_any = true;
                w.symbols.swap(merged_syms);
            }
        }
        if (!changed_any)
        {
            break;
        }

        ++merges_done;
        if (merges_done % 1000 == 0)
        {
            std::cerr << "WordPiece merges: " << merges_done << " vocab=" << id_to_symbol.size() << "/" << target_vocab
                      << "\n";
        }
    }
}
} // namespace

bool train_backend_wordpiece(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                             std::string &err)
{
    (void)err;
    auto specials = make_special_tokens(cfg);
    std::vector<WPWord> words;
    words.reserve(global_counts.size());
    SymbolMap symbol_to_id;
    symbol_to_id.reserve(8192);
    std::vector<std::string> id_to_symbol;
    id_to_symbol.reserve(8192);

    for (const auto &kv : global_counts)
    {
        if (kv.second < cfg.min_freq)
        {
            continue;
        }
        auto cps = split_codepoints(kv.first);
        if (cps.empty())
        {
            continue;
        }
        WPWord word;
        word.freq = kv.second;
        word.symbols.reserve(cps.size());
        for (std::size_t i = 0; i < cps.size(); ++i)
        {
            std::string tok = (i == 0) ? cps[i] : (cfg.wordpiece_continuing_prefix + cps[i]);
            word.symbols.push_back(ensure_symbol(tok, symbol_to_id, id_to_symbol));
        }
        words.push_back(std::move(word));
    }

    std::size_t target_vocab = calc_pair_target_vocab(cfg, id_to_symbol.size(), specials.size());
    train_wordpiece(words, id_to_symbol, symbol_to_id, target_vocab, cfg.min_pair_freq,
                    cfg.wordpiece_continuing_prefix);

    append_symbols_to_vocab(specials, id_to_symbol, artifacts.id_to_token);
    artifacts.merges.clear();
    artifacts.has_merges = false;
    artifacts.token_scores.assign(artifacts.id_to_token.size(), -1.0);
    return true;
}
