#include "train_backend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "train_backend_common.h"

namespace
{
struct UnigramWord
{
    std::vector<std::string> cps;
    uint64_t freq = 0;
};

struct UniToken
{
    std::string token;
    std::vector<std::string> cps;
    double weight = 0.0;
    double logp = 0.0;
    bool required = false;
};

std::unordered_map<std::string, uint64_t> build_unigram_seed_counts(const GlobalCountMap &global_counts,
                                                                    std::unordered_set<std::string> &required_chars,
                                                                    std::size_t min_freq, std::size_t max_token_length)
{
    std::unordered_map<std::string, uint64_t> counts;
    counts.reserve(global_counts.size() * 4 + 1024);
    for (const auto &kv : global_counts)
    {
        if (kv.second < min_freq)
        {
            continue;
        }
        auto cps = split_codepoints(kv.first);
        if (cps.empty())
        {
            continue;
        }
        for (const auto &cp : cps)
        {
            required_chars.insert(cp);
            counts[cp] += kv.second;
        }
        std::size_t n = cps.size();
        if (n < 2)
        {
            continue;
        }
        std::size_t lim = std::min<std::size_t>(n, max_token_length);
        for (std::size_t i = 0; i < n; ++i)
        {
            std::string token = cps[i];
            for (std::size_t len = 2; len <= lim && i + len <= n; ++len)
            {
                token += cps[i + len - 1];
                counts[token] += kv.second;
            }
        }
    }
    return counts;
}

std::vector<UnigramWord> build_unigram_words(const GlobalCountMap &global_counts, std::size_t min_freq)
{
    std::vector<UnigramWord> words;
    words.reserve(global_counts.size());
    for (const auto &kv : global_counts)
    {
        if (kv.second < min_freq)
        {
            continue;
        }
        auto cps = split_codepoints(kv.first);
        if (cps.empty())
        {
            continue;
        }
        words.push_back({std::move(cps), kv.second});
    }
    return words;
}

void rebuild_unigram_index(const std::vector<UniToken> &tokens,
                           std::unordered_map<std::string, std::vector<int>> &index)
{
    index.clear();
    index.reserve(tokens.size() / 2 + 16);
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        if (tokens[i].cps.empty())
        {
            continue;
        }
        index[tokens[i].cps.front()].push_back(static_cast<int>(i));
    }
    for (auto &kv : index)
    {
        auto &vec = kv.second;
        std::sort(vec.begin(), vec.end(), [&](int a, int b) { return tokens[a].cps.size() > tokens[b].cps.size(); });
    }
}

bool token_matches(const UniToken &token, const std::vector<std::string> &cps, std::size_t pos)
{
    if (pos + token.cps.size() > cps.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < token.cps.size(); ++i)
    {
        if (token.cps[i] != cps[pos + i])
        {
            return false;
        }
    }
    return true;
}

std::vector<int> unigram_best_segmentation(const std::vector<std::string> &cps, const std::vector<UniToken> &tokens,
                                           const std::unordered_map<std::string, std::vector<int>> &index)
{
    const double neg_inf = -std::numeric_limits<double>::infinity();
    std::size_t n = cps.size();
    std::vector<double> best(n + 1, neg_inf);
    std::vector<int> prev_pos(n + 1, -1);
    std::vector<int> prev_tok(n + 1, -1);
    best[0] = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(best[i]))
        {
            continue;
        }
        auto it = index.find(cps[i]);
        if (it == index.end())
        {
            continue;
        }
        for (int tok_idx : it->second)
        {
            const auto &tok = tokens[tok_idx];
            if (!token_matches(tok, cps, i))
            {
                continue;
            }
            std::size_t j = i + tok.cps.size();
            double cand = best[i] + tok.logp;
            if (cand > best[j])
            {
                best[j] = cand;
                prev_pos[j] = static_cast<int>(i);
                prev_tok[j] = tok_idx;
            }
        }
    }

    if (!std::isfinite(best[n]))
    {
        std::vector<int> fallback;
        fallback.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            auto it = index.find(cps[i]);
            if (it == index.end() || it->second.empty())
            {
                continue;
            }
            fallback.push_back(it->second.back());
        }
        return fallback;
    }

    std::vector<int> out;
    int cur = static_cast<int>(n);
    while (cur > 0)
    {
        int tok_idx = prev_tok[static_cast<std::size_t>(cur)];
        int p = prev_pos[static_cast<std::size_t>(cur)];
        if (tok_idx < 0 || p < 0)
        {
            break;
        }
        out.push_back(tok_idx);
        cur = p;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

void normalize_unigram_weights(std::vector<UniToken> &tokens)
{
    double total = 0.0;
    for (auto &tok : tokens)
    {
        if (!(tok.weight > 0.0))
        {
            tok.weight = 1e-12;
        }
        total += tok.weight;
    }
    if (!(total > 0.0))
    {
        total = static_cast<double>(tokens.size());
    }
    for (auto &tok : tokens)
    {
        double p = tok.weight / total;
        if (!(p > 0.0))
        {
            p = 1e-12;
        }
        tok.logp = std::log(p);
    }
}

void prune_unigram_tokens(std::vector<UniToken> &tokens, std::size_t keep_size)
{
    if (tokens.size() <= keep_size)
    {
        return;
    }
    std::vector<int> idx(tokens.size());
    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        idx[i] = static_cast<int>(i);
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (tokens[a].required != tokens[b].required)
        {
            return tokens[a].required > tokens[b].required;
        }
        if (tokens[a].weight != tokens[b].weight)
        {
            return tokens[a].weight > tokens[b].weight;
        }
        return tokens[a].token < tokens[b].token;
    });

    std::vector<UniToken> kept;
    kept.reserve(keep_size);
    for (int id : idx)
    {
        if (kept.size() >= keep_size && !tokens[id].required)
        {
            continue;
        }
        kept.push_back(std::move(tokens[id]));
    }
    tokens.swap(kept);
}
} // namespace

bool train_backend_unigram(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                           std::string &err)
{
    auto specials = make_special_tokens(cfg);
    std::unordered_set<std::string> required_chars;
    auto seed_counts = build_unigram_seed_counts(global_counts, required_chars, cfg.min_freq, cfg.max_token_length);
    auto words = build_unigram_words(global_counts, cfg.min_freq);
    if (words.empty())
    {
        err = "no words survived min_freq for unigram";
        return false;
    }

    std::size_t special_count = specials.size();
    std::size_t target_model_vocab = 0;
    if (cfg.vocab_size > special_count)
    {
        target_model_vocab = cfg.vocab_size - special_count;
    }
    target_model_vocab = std::max<std::size_t>(target_model_vocab, required_chars.size());
    if (target_model_vocab == 0)
    {
        target_model_vocab = required_chars.size();
    }

    std::size_t seed_target =
        std::max<std::size_t>(target_model_vocab * cfg.unigram_seed_multiplier, required_chars.size() + 32);
    std::vector<std::pair<std::string, uint64_t>> candidates;
    candidates.reserve(seed_counts.size());
    for (const auto &kv : seed_counts)
    {
        candidates.emplace_back(kv.first, kv.second);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });

    std::vector<UniToken> tokens;
    tokens.reserve(std::min<std::size_t>(seed_target, candidates.size()) + required_chars.size());
    std::unordered_set<std::string> seen;
    seen.reserve(seed_target * 2 + 32);

    for (const auto &cp : required_chars)
    {
        UniToken tok;
        tok.token = cp;
        tok.cps = {cp};
        tok.required = true;
        auto it = seed_counts.find(cp);
        tok.weight = (it != seed_counts.end() ? static_cast<double>(it->second) : 1.0) + 1e-6;
        tokens.push_back(std::move(tok));
        seen.insert(cp);
    }
    for (const auto &kv : candidates)
    {
        if (tokens.size() >= seed_target)
        {
            break;
        }
        if (!seen.insert(kv.first).second)
        {
            continue;
        }
        UniToken tok;
        tok.token = kv.first;
        tok.cps = split_codepoints(kv.first);
        tok.required = required_chars.find(kv.first) != required_chars.end();
        tok.weight = static_cast<double>(kv.second) + 1e-6;
        tokens.push_back(std::move(tok));
    }

    if (tokens.empty())
    {
        err = "failed to build unigram seed vocabulary";
        return false;
    }

    normalize_unigram_weights(tokens);

    std::unordered_map<std::string, std::vector<int>> index;
    for (std::size_t iter = 0; iter < cfg.unigram_em_iters; ++iter)
    {
        rebuild_unigram_index(tokens, index);
        std::vector<double> counts(tokens.size(), 0.0);
        for (const auto &word : words)
        {
            auto seg = unigram_best_segmentation(word.cps, tokens, index);
            for (int tid : seg)
            {
                if (tid >= 0 && static_cast<std::size_t>(tid) < counts.size())
                {
                    counts[static_cast<std::size_t>(tid)] += static_cast<double>(word.freq);
                }
            }
        }
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            if (tokens[i].required)
            {
                counts[i] += 1e-3;
            }
            tokens[i].weight = counts[i] + 1e-12;
        }
        normalize_unigram_weights(tokens);

        std::size_t floor_keep = std::max<std::size_t>(target_model_vocab, required_chars.size());
        std::size_t keep_size = floor_keep;
        if (iter + 1 < cfg.unigram_em_iters)
        {
            keep_size =
                static_cast<std::size_t>(std::ceil(static_cast<double>(tokens.size()) * cfg.unigram_prune_ratio));
            keep_size = std::max<std::size_t>(keep_size, floor_keep);
        }
        prune_unigram_tokens(tokens, keep_size);
        std::cerr << "Unigram iter " << (iter + 1) << "/" << cfg.unigram_em_iters << " vocab=" << tokens.size() << "\n";
    }

    prune_unigram_tokens(tokens, std::max<std::size_t>(target_model_vocab, required_chars.size()));
    normalize_unigram_weights(tokens);

    std::sort(tokens.begin(), tokens.end(), [](const UniToken &a, const UniToken &b) {
        if (a.required != b.required)
        {
            return a.required > b.required;
        }
        if (a.weight != b.weight)
        {
            return a.weight > b.weight;
        }
        return a.token < b.token;
    });

    std::vector<std::string> model_tokens;
    std::vector<double> model_scores;
    model_tokens.reserve(tokens.size());
    model_scores.reserve(tokens.size());
    for (const auto &tok : tokens)
    {
        model_tokens.push_back(tok.token);
        model_scores.push_back(tok.logp);
    }

    artifacts.id_to_token = specials;
    artifacts.token_scores.assign(specials.size(), -100.0);
    for (std::size_t i = 0; i < model_tokens.size(); ++i)
    {
        if (std::find(artifacts.id_to_token.begin(), artifacts.id_to_token.end(), model_tokens[i]) !=
            artifacts.id_to_token.end())
        {
            continue;
        }
        artifacts.id_to_token.push_back(model_tokens[i]);
        artifacts.token_scores.push_back(model_scores[i]);
    }
    artifacts.merges.clear();
    artifacts.has_merges = false;
    return true;
}
