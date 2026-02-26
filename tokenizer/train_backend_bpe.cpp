#include "train_backend.h"

#include <unordered_map>
#include <vector>

#include "tokenflux_bpe.h"
#include "tokenflux_lib.h"
#include "train_backend_common.h"

bool train_backend_bpe(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                       std::string &err)
{
    (void)err;
    auto specials = make_special_tokens(cfg);
    std::unordered_map<uint32_t, int> cp_to_id;
    cp_to_id.reserve(4096);
    std::vector<std::string> id_to_symbol;
    id_to_symbol.reserve(8192);

    for (const auto &kv : global_counts)
    {
        std::size_t i = 0;
        while (i < kv.first.size())
        {
            uint32_t cp = 0;
            if (!next_codepoint(kv.first, i, cp))
            {
                break;
            }
            if (cp_to_id.find(cp) != cp_to_id.end())
            {
                continue;
            }
            std::string cp_utf8;
            append_utf8(cp, cp_utf8);
            int id = static_cast<int>(id_to_symbol.size());
            id_to_symbol.push_back(cp_utf8);
            cp_to_id.emplace(cp, id);
        }
    }

    auto words = build_words_from_tokens(global_counts, cp_to_id, cfg.min_freq);
    std::size_t target_vocab = calc_pair_target_vocab(cfg, id_to_symbol.size(), specials.size());
    std::vector<std::string> merges;
    merges.reserve(target_vocab > id_to_symbol.size() ? (target_vocab - id_to_symbol.size()) : 0);
    train_bpe(words, id_to_symbol, merges, target_vocab, cfg.min_pair_freq, cfg.pair_max_entries);

    append_symbols_to_vocab(specials, id_to_symbol, artifacts.id_to_token);
    artifacts.merges = std::move(merges);
    artifacts.has_merges = true;
    artifacts.token_scores.assign(artifacts.id_to_token.size(), -1.0);
    return true;
}
