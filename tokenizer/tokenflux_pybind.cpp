#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "input_source.hpp"
#include "tokenflux_config.hpp"
#include "tokenflux_lib.hpp"
#include "tokenize_common.hpp"
#include "tokenize_pipeline.hpp"
#include "tokenize_tokenizer.hpp"
#include "train_pipeline.hpp"

namespace py = pybind11;

namespace
{
class PythonTokenizer
{
  public:
    PythonTokenizer() = default;
    explicit PythonTokenizer(const std::string &path)
    {
        load(path);
    }

    void load(const std::string &path)
    {
        std::string err;
        if (!encoder_.load(path, err))
        {
            throw std::runtime_error(err);
        }
        tokenizer_path_ = path;
        cache_.clear();
    }

    const std::string &tokenizer_path() const
    {
        return tokenizer_path_;
    }

    std::size_t vocab_size() const
    {
        return encoder_.vocab_size();
    }

    std::string model_name() const
    {
        return encoder_.model_name();
    }

    py::object token_to_id(const std::string &token) const
    {
        std::uint32_t id = 0;
        if (!encoder_.token_to_id(token, id))
        {
            return py::none();
        }
        return py::int_(id);
    }

    std::vector<std::uint32_t> encode(const std::string &text, const std::string &bos_token,
                                      const std::string &eos_token, bool reset_cache)
    {
        if (reset_cache)
        {
            cache_.clear();
        }
        std::int64_t bos_id = resolve_optional_token_id(bos_token, "bos");
        std::int64_t eos_id = resolve_optional_token_id(eos_token, "eos");
        return encode_impl(text, bos_id, eos_id, cache_);
    }

    std::vector<std::vector<std::uint32_t>> encode_batch(const std::vector<std::string> &texts, const std::string &bos_token,
                                                         const std::string &eos_token, bool reset_cache)
    {
        if (reset_cache)
        {
            cache_.clear();
        }
        std::int64_t bos_id = resolve_optional_token_id(bos_token, "bos");
        std::int64_t eos_id = resolve_optional_token_id(eos_token, "eos");
        std::vector<std::vector<std::uint32_t>> out;
        out.reserve(texts.size());
        for (const auto &text : texts)
        {
            out.push_back(encode_impl(text, bos_id, eos_id, cache_));
        }
        return out;
    }

    py::object encode_to_torch(const std::string &text, const std::string &bos_token, const std::string &eos_token,
                               const std::string &dtype, bool reset_cache)
    {
        auto ids = encode(text, bos_token, eos_token, reset_cache);
        return make_torch_tensor_u32(ids, dtype);
    }

    py::dict encode_batch_to_torch(const std::vector<std::string> &texts, const std::string &bos_token,
                                   const std::string &eos_token, std::int64_t pad_id, const std::string &dtype,
                                   bool reset_cache)
    {
        auto batch = encode_batch(texts, bos_token, eos_token, reset_cache);
        std::size_t max_len = 0;
        for (const auto &ids : batch)
        {
            max_len = std::max(max_len, ids.size());
        }

        py::list rows;
        std::vector<std::int64_t> lengths;
        lengths.reserve(batch.size());
        for (const auto &ids : batch)
        {
            py::list row;
            for (auto id : ids)
            {
                row.append(py::int_(id));
            }
            for (std::size_t i = ids.size(); i < max_len; ++i)
            {
                row.append(py::int_(pad_id));
            }
            rows.append(std::move(row));
            lengths.push_back(static_cast<std::int64_t>(ids.size()));
        }

        py::module_ torch = py::module_::import("torch");
        py::object dtype_obj = resolve_torch_dtype(torch, dtype);

        py::dict out;
        out["input_ids"] = torch.attr("tensor")(rows, py::arg("dtype") = dtype_obj);
        out["lengths"] = torch.attr("tensor")(py::cast(lengths), py::arg("dtype") = torch.attr("int64"));
        return out;
    }

    py::dict tokenize_inputs_to_torch(const std::vector<std::string> &inputs, const std::string &text_field,
                                      std::size_t min_chars, std::size_t max_chars, const std::string &bos_token,
                                      const std::string &eos_token, const std::string &dtype, bool reset_cache)
    {
        if (reset_cache)
        {
            cache_.clear();
        }

        std::int64_t bos_id = resolve_optional_token_id(bos_token, "bos");
        std::int64_t eos_id = resolve_optional_token_id(eos_token, "eos");

        std::vector<std::uint32_t> flat_ids;
        std::vector<std::int64_t> doc_offsets;
        std::vector<std::int64_t> doc_lengths;
        std::vector<std::string> source_names;
        std::uint64_t num_skipped = 0;

        std::vector<InputSource> sources;
        std::string err;
        auto remote_cache_dir = (std::filesystem::temp_directory_path() / "tokenflux_cpp" / "remote_inputs").string();
        if (!resolve_input_sources(inputs, "", "", remote_cache_dir, sources, err))
        {
            throw std::runtime_error(err);
        }

        doc_offsets.push_back(0);
        for (const auto &source : sources)
        {
            std::string local_err;
            bool ok = for_each_text_record(
                source.local_path, text_field,
                [&](const std::string &incoming_text) {
                    if (incoming_text.empty())
                    {
                        return;
                    }
                    std::string text = incoming_text;
                    std::size_t chars = utf8_char_count(text);
                    if (chars < min_chars)
                    {
                        ++num_skipped;
                        return;
                    }
                    if (max_chars > 0 && chars > max_chars)
                    {
                        text = truncate_utf8(text, max_chars);
                    }
                    auto ids = encode_impl(text, bos_id, eos_id, cache_);
                    flat_ids.insert(flat_ids.end(), ids.begin(), ids.end());
                    doc_lengths.push_back(static_cast<std::int64_t>(ids.size()));
                    doc_offsets.push_back(static_cast<std::int64_t>(flat_ids.size()));
                    source_names.push_back(source.source);
                },
                local_err);
            if (!ok)
            {
                throw std::runtime_error(local_err.empty() ? ("failed to read input file: " + source.source) : local_err);
            }
        }

        py::dict out;
        out["token_ids"] = make_torch_tensor_u32(flat_ids, dtype);
        out["doc_offsets"] = make_torch_tensor_i64(doc_offsets);
        out["doc_lengths"] = make_torch_tensor_i64(doc_lengths);
        out["num_docs"] = py::int_(doc_lengths.size());
        out["num_skipped"] = py::int_(num_skipped);
        out["sources"] = py::cast(source_names);
        return out;
    }

  private:
    static std::size_t utf8_char_count(const std::string &text)
    {
        std::size_t i = 0;
        std::size_t n = 0;
        while (i < text.size())
        {
            std::size_t prev = i;
            std::uint32_t cp = 0;
            if (!next_codepoint(text, i, cp))
            {
                break;
            }
            if (i <= prev)
            {
                break;
            }
            ++n;
        }
        return n;
    }

    static py::object resolve_torch_dtype(const py::module_ &torch, const std::string &dtype)
    {
        std::string key = dtype;
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (key == "int32")
        {
            return torch.attr("int32");
        }
        if (key == "int64" || key == "long")
        {
            return torch.attr("int64");
        }
        if (key == "int16" || key == "short")
        {
            return torch.attr("int16");
        }
        if (key == "uint8")
        {
            return torch.attr("uint8");
        }
        throw std::runtime_error("unsupported torch dtype: " + dtype);
    }

    static py::object make_torch_tensor_i64(const std::vector<std::int64_t> &values)
    {
        py::module_ torch = py::module_::import("torch");
        return torch.attr("tensor")(py::cast(values), py::arg("dtype") = torch.attr("int64"));
    }

    static py::object make_torch_tensor_u32(const std::vector<std::uint32_t> &values, const std::string &dtype)
    {
        py::module_ torch = py::module_::import("torch");
        return torch.attr("tensor")(py::cast(values), py::arg("dtype") = resolve_torch_dtype(torch, dtype));
    }

    std::int64_t resolve_optional_token_id(const std::string &token, const char *kind) const
    {
        if (token.empty())
        {
            return -1;
        }
        std::uint32_t id = 0;
        if (!encoder_.token_to_id(token, id))
        {
            throw std::runtime_error(std::string(kind) + " token not found in tokenizer: " + token);
        }
        return static_cast<std::int64_t>(id);
    }

    std::vector<std::uint32_t> encode_impl(const std::string &text, std::int64_t bos_id, std::int64_t eos_id,
                                           std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
    {
        std::vector<std::uint32_t> out;
        if (bos_id >= 0)
        {
            out.push_back(static_cast<std::uint32_t>(bos_id));
        }
        encoder_.encode_text_append(text, cache, out);
        if (eos_id >= 0)
        {
            out.push_back(static_cast<std::uint32_t>(eos_id));
        }
        return out;
    }

    tokenflux::tokenize::TokenizerEncoder encoder_;
    std::unordered_map<std::string, std::vector<std::uint32_t>> cache_;
    std::string tokenizer_path_;
};
} // namespace

PYBIND11_MODULE(tokenflux_cpp, m)
{
    m.doc() = "TokenFlux C++ bindings";
    m.attr("__version__") = "0.3.2";

    py::enum_<TrainerKind>(m, "TrainerKind")
        .value("byte_bpe", TrainerKind::byte_bpe)
        .value("bpe", TrainerKind::bpe)
        .value("wordpiece", TrainerKind::wordpiece)
        .value("unigram", TrainerKind::unigram);

    py::class_<Config>(m, "TrainConfig")
        .def(py::init<>())
        .def_readwrite("env_path", &Config::env_path)
        .def_readwrite("data_glob", &Config::data_glob)
        .def_readwrite("data_list", &Config::data_list)
        .def_readwrite("input_entries", &Config::input_entries)
        .def_readwrite("text_field", &Config::text_field)
        .def_readwrite("output_json", &Config::output_json)
        .def_readwrite("output_vocab", &Config::output_vocab)
        .def_readwrite("output_merges", &Config::output_merges)
        .def_readwrite("unk_token", &Config::unk_token)
        .def_readwrite("special_tokens", &Config::special_tokens)
        .def_readwrite("trainer", &Config::trainer)
        .def_readwrite("vocab_size", &Config::vocab_size)
        .def_readwrite("min_freq", &Config::min_freq)
        .def_readwrite("min_pair_freq", &Config::min_pair_freq)
        .def_readwrite("chunk_files", &Config::chunk_files)
        .def_readwrite("chunk_docs", &Config::chunk_docs)
        .def_readwrite("top_k", &Config::top_k)
        .def_readwrite("max_chars_per_doc", &Config::max_chars_per_doc)
        .def_readwrite("threads", &Config::threads)
        .def_readwrite("progress_interval_ms", &Config::progress_interval_ms)
        .def_readwrite("max_memory_mb", &Config::max_memory_mb)
        .def_readwrite("pair_max_entries", &Config::pair_max_entries)
        .def_readwrite("records_per_chunk", &Config::records_per_chunk)
        .def_readwrite("queue_capacity", &Config::queue_capacity)
        .def_readwrite("max_token_length", &Config::max_token_length)
        .def_readwrite("unigram_em_iters", &Config::unigram_em_iters)
        .def_readwrite("unigram_seed_multiplier", &Config::unigram_seed_multiplier)
        .def_readwrite("unigram_prune_ratio", &Config::unigram_prune_ratio)
        .def_readwrite("wordpiece_continuing_prefix", &Config::wordpiece_continuing_prefix)
        .def_readwrite("prescan_records", &Config::prescan_records)
        .def_readwrite("chunk_dir", &Config::chunk_dir)
        .def_readwrite("resume", &Config::resume)
        .def_readwrite("write_vocab", &Config::write_vocab)
        .def_readwrite("write_merges", &Config::write_merges);

    py::class_<tokenflux::tokenize::Args>(m, "TokenizeArgs")
        .def(py::init<>())
        .def_readwrite("env_file", &tokenflux::tokenize::Args::env_file)
        .def_readwrite("data_glob", &tokenflux::tokenize::Args::data_glob)
        .def_readwrite("data_list", &tokenflux::tokenize::Args::data_list)
        .def_readwrite("input_entries", &tokenflux::tokenize::Args::input_entries)
        .def_readwrite("text_field", &tokenflux::tokenize::Args::text_field)
        .def_readwrite("tokenizer_path", &tokenflux::tokenize::Args::tokenizer_path)
        .def_readwrite("out_dir", &tokenflux::tokenize::Args::out_dir)
        .def_readwrite("max_tokens_per_shard", &tokenflux::tokenize::Args::max_tokens_per_shard)
        .def_readwrite("encode_batch_size", &tokenflux::tokenize::Args::encode_batch_size)
        .def_readwrite("min_chars", &tokenflux::tokenize::Args::min_chars)
        .def_readwrite("max_chars", &tokenflux::tokenize::Args::max_chars)
        .def_readwrite("max_docs", &tokenflux::tokenize::Args::max_docs)
        .def_readwrite("eos_token", &tokenflux::tokenize::Args::eos_token)
        .def_readwrite("bos_token", &tokenflux::tokenize::Args::bos_token)
        .def_readwrite("progress_every", &tokenflux::tokenize::Args::progress_every)
        .def_readwrite("threads", &tokenflux::tokenize::Args::threads)
        .def_readwrite("cache_max_entries", &tokenflux::tokenize::Args::cache_max_entries)
        .def_readwrite("max_memory_mb", &tokenflux::tokenize::Args::max_memory_mb)
        .def_readwrite("prescan_records", &tokenflux::tokenize::Args::prescan_records)
        .def_readwrite("resume", &tokenflux::tokenize::Args::resume);

    py::class_<PythonTokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def(py::init<const std::string &>(), py::arg("tokenizer_path"))
        .def("load", &PythonTokenizer::load, py::arg("tokenizer_path"))
        .def_property_readonly("tokenizer_path", &PythonTokenizer::tokenizer_path)
        .def_property_readonly("vocab_size", &PythonTokenizer::vocab_size)
        .def_property_readonly("model_name", &PythonTokenizer::model_name)
        .def("token_to_id", &PythonTokenizer::token_to_id, py::arg("token"))
        .def("encode", &PythonTokenizer::encode, py::arg("text"), py::arg("bos_token") = "",
             py::arg("eos_token") = "", py::arg("reset_cache") = false)
        .def("encode_batch", &PythonTokenizer::encode_batch, py::arg("texts"), py::arg("bos_token") = "",
             py::arg("eos_token") = "", py::arg("reset_cache") = false)
        .def("encode_to_torch", &PythonTokenizer::encode_to_torch, py::arg("text"), py::arg("bos_token") = "",
             py::arg("eos_token") = "", py::arg("dtype") = "int64", py::arg("reset_cache") = false)
        .def("encode_batch_to_torch", &PythonTokenizer::encode_batch_to_torch, py::arg("texts"),
             py::arg("bos_token") = "", py::arg("eos_token") = "", py::arg("pad_id") = 0, py::arg("dtype") = "int64",
             py::arg("reset_cache") = false)
        .def("tokenize_inputs_to_torch", &PythonTokenizer::tokenize_inputs_to_torch, py::arg("inputs"),
             py::arg("text_field") = "text", py::arg("min_chars") = 1, py::arg("max_chars") = 20000,
             py::arg("bos_token") = "", py::arg("eos_token") = "", py::arg("dtype") = "int64",
             py::arg("reset_cache") = false);

    m.def("train",
          [](Config cfg, const std::vector<std::string> &inputs) {
              cfg.input_entries = inputs;
              py::gil_scoped_release release;
              int rc = run_train(std::move(cfg));
              if (rc != 0)
              {
                  throw std::runtime_error("TokenFlux train failed");
              }
          },
          py::arg("config"), py::arg("inputs") = std::vector<std::string>{});

    m.def("tokenize",
          [](tokenflux::tokenize::Args args, const std::vector<std::string> &inputs) {
              args.input_entries = inputs;
              py::gil_scoped_release release;
              int rc = tokenflux::tokenize::run_tokenize(args);
              if (rc != 0)
              {
                  throw std::runtime_error("TokenFlux tokenize failed");
              }
          },
          py::arg("args"), py::arg("inputs") = std::vector<std::string>{});
}
