#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tokenflux_config.hpp"
#include "tokenize_common.hpp"
#include "tokenize_pipeline.hpp"
#include "train_pipeline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tokenflux_cpp, m)
{
    m.doc() = "TokenFlux C++ bindings";
    m.attr("__version__") = "0.3.0";

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

    m.def("train", [](Config cfg) {
        int rc = run_train(std::move(cfg));
        if (rc != 0)
        {
            throw std::runtime_error("TokenFlux train failed");
        }
    });

    m.def("tokenize", [](tokenflux::tokenize::Args args) {
        int rc = tokenflux::tokenize::run_tokenize(args);
        if (rc != 0)
        {
            throw std::runtime_error("TokenFlux tokenize failed");
        }
    });
}
