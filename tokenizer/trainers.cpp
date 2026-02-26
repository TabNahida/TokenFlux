#include "trainers.hpp"

#include "train_backend.hpp"
#include "train_backend_common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>

#include "tokenflux_lib.hpp"

namespace
{
bool write_generic_tokenizer_json(const Config &cfg, const TrainArtifacts &artifacts, std::string &err)
{
    std::ofstream out(cfg.output_json, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        err = "failed to write tokenizer.json: " + cfg.output_json;
        return false;
    }
    auto specials = make_special_tokens(cfg);
    auto is_special = [&](const std::string &tok) -> bool {
        return std::find(specials.begin(), specials.end(), tok) != specials.end();
    };
    auto unk_it = std::find(artifacts.id_to_token.begin(), artifacts.id_to_token.end(), cfg.unk_token);
    int unk_id = unk_it == artifacts.id_to_token.end() ? 0 : static_cast<int>(unk_it - artifacts.id_to_token.begin());

    out << "{";
    out << "\"version\":\"1.0\",";
    out << "\"truncation\":null,";
    out << "\"padding\":null,";

    out << "\"added_tokens\":[";
    bool first_added = true;
    for (std::size_t i = 0; i < artifacts.id_to_token.size(); ++i)
    {
        const auto &tok = artifacts.id_to_token[i];
        if (!is_special(tok))
        {
            continue;
        }
        if (!first_added)
        {
            out << ",";
        }
        first_added = false;
        out << "{";
        out << "\"id\":" << i << ",";
        out << "\"content\":\"" << json_escape(tok) << "\",";
        out << "\"single_word\":false,";
        out << "\"lstrip\":false,";
        out << "\"rstrip\":false,";
        out << "\"normalized\":false,";
        out << "\"special\":true";
        out << "}";
    }
    out << "],";

    out << "\"normalizer\":null,";
    out << "\"pre_tokenizer\":{\"type\":\"WhitespaceSplit\"},";
    out << "\"post_processor\":null,";

    if (cfg.trainer == TrainerKind::wordpiece)
    {
        out << "\"decoder\":{\"type\":\"WordPiece\",\"prefix\":\"" << json_escape(cfg.wordpiece_continuing_prefix)
            << "\",\"cleanup\":true},";
        out << "\"model\":{";
        out << "\"type\":\"WordPiece\",";
        out << "\"unk_token\":\"" << json_escape(cfg.unk_token) << "\",";
        out << "\"continuing_subword_prefix\":\"" << json_escape(cfg.wordpiece_continuing_prefix) << "\",";
        out << "\"max_input_chars_per_word\":100,";
        out << "\"vocab\":{";
        for (std::size_t i = 0; i < artifacts.id_to_token.size(); ++i)
        {
            if (i > 0)
            {
                out << ",";
            }
            out << "\"" << json_escape(artifacts.id_to_token[i]) << "\":" << i;
        }
        out << "}";
        out << "}";
    }
    else if (cfg.trainer == TrainerKind::unigram)
    {
        out << "\"decoder\":null,";
        out << "\"model\":{";
        out << "\"type\":\"Unigram\",";
        out << "\"unk_id\":" << unk_id << ",";
        out << "\"byte_fallback\":false,";
        out << "\"vocab\":[";
        for (std::size_t i = 0; i < artifacts.id_to_token.size(); ++i)
        {
            if (i > 0)
            {
                out << ",";
            }
            double score = (i < artifacts.token_scores.size()) ? artifacts.token_scores[i] : -10.0;
            out << "[\"" << json_escape(artifacts.id_to_token[i]) << "\"," << std::setprecision(8) << score << "]";
        }
        out << "]";
        out << "}";
    }
    else
    {
        out << "\"decoder\":null,";
        out << "\"model\":{";
        out << "\"type\":\"BPE\",";
        out << "\"dropout\":null,";
        out << "\"unk_token\":\"" << json_escape(cfg.unk_token) << "\",";
        out << "\"continuing_subword_prefix\":\"\",";
        out << "\"end_of_word_suffix\":\"\",";
        out << "\"fuse_unk\":false,";
        out << "\"vocab\":{";
        for (std::size_t i = 0; i < artifacts.id_to_token.size(); ++i)
        {
            if (i > 0)
            {
                out << ",";
            }
            out << "\"" << json_escape(artifacts.id_to_token[i]) << "\":" << i;
        }
        out << "},";
        out << "\"merges\":[";
        for (std::size_t i = 0; i < artifacts.merges.size(); ++i)
        {
            if (i > 0)
            {
                out << ",";
            }
            const auto &m = artifacts.merges[i];
            auto sp = m.find(' ');
            if (sp == std::string::npos)
            {
                out << "[\"" << json_escape(m) << "\",\"\"]";
            }
            else
            {
                out << "[\"" << json_escape(m.substr(0, sp)) << "\",\"" << json_escape(m.substr(sp + 1)) << "\"]";
            }
        }
        out << "]";
        out << "}";
    }
    out << "}";
    if (!out)
    {
        err = "failed to flush tokenizer.json: " + cfg.output_json;
        return false;
    }
    return true;
}
} // namespace

ProcessTextFn build_process_text_callback(const Config &cfg)
{
    auto byte_to_unicode_cp = build_byte_to_unicode_cp();
    auto byte_to_unicode = build_byte_to_unicode_str(byte_to_unicode_cp);
    return [cfg, byte_to_unicode](const std::string &text, LocalCountMap &local_counts, uint64_t &doc_count,
                                  std::size_t &reduce_counter, std::size_t local_entry_cap) {
        std::string trimmed = text;
        if (cfg.max_chars_per_doc > 0)
        {
            trimmed = truncate_utf8(trimmed, cfg.max_chars_per_doc);
        }

        if (cfg.trainer == TrainerKind::byte_bpe)
        {
            auto tokens = pretokenize(trimmed);
            for (const auto &tok : tokens)
            {
                if (tok.empty())
                {
                    continue;
                }
                std::string encoded = byte_level_encode(tok, byte_to_unicode);
                if (!encoded.empty())
                {
                    ++local_counts[encoded];
                }
            }
        }
        else
        {
            auto words = split_whitespace_words(trimmed);
            for (const auto &word : words)
            {
                if (!word.empty())
                {
                    ++local_counts[word];
                }
            }
        }

        ++doc_count;
        maybe_reduce_local(local_counts, cfg, reduce_counter, local_entry_cap);
    };
}

bool train_from_global_counts(const Config &cfg, const GlobalCountMap &global_counts, TrainArtifacts &artifacts,
                              std::string &err)
{
    artifacts = {};
    if (global_counts.empty())
    {
        err = "global counts are empty";
        return false;
    }

    switch (cfg.trainer)
    {
    case TrainerKind::byte_bpe:
        return train_backend_byte_bpe(cfg, global_counts, artifacts, err);
    case TrainerKind::bpe:
        return train_backend_bpe(cfg, global_counts, artifacts, err);
    case TrainerKind::wordpiece:
        return train_backend_wordpiece(cfg, global_counts, artifacts, err);
    case TrainerKind::unigram:
        return train_backend_unigram(cfg, global_counts, artifacts, err);
    }
    err = "unsupported trainer type";
    return false;
}

bool write_trained_tokenizer(const Config &cfg, const TrainArtifacts &artifacts, std::string &err)
{
    if (cfg.trainer == TrainerKind::byte_bpe)
    {
        if (!write_tokenizer_json(cfg.output_json, cfg, artifacts.id_to_token, artifacts.merges))
        {
            err = "failed to write tokenizer.json: " + cfg.output_json;
            return false;
        }
    }
    else
    {
        if (!write_generic_tokenizer_json(cfg, artifacts, err))
        {
            return false;
        }
    }

    if (cfg.write_vocab)
    {
        if (!write_vocab_json(cfg.output_vocab, artifacts.id_to_token))
        {
            err = "failed to write vocab.json: " + cfg.output_vocab;
            return false;
        }
    }

    if (cfg.write_merges && artifacts.has_merges)
    {
        if (!write_merges_txt(cfg.output_merges, artifacts.merges))
        {
            err = "failed to write merges.txt: " + cfg.output_merges;
            return false;
        }
    }
    return true;
}
