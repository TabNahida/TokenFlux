#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tokenflux/formats.hpp"
#include "tokenflux/pretokenize.hpp"
#include "tokenflux/stream.hpp"
#include "tokenflux/tokenizer.hpp"
#include "tokenflux/trainer.hpp"

namespace py = pybind11;
using namespace tokenflux;

PYBIND11_MODULE(pytokenflux, m) {
  py::class_<EncodeOptions>(m, "EncodeOptions")
      .def(py::init<>())
      .def_readwrite("add_bos", &EncodeOptions::add_bos)
      .def_readwrite("add_eos", &EncodeOptions::add_eos);

  py::class_<DecodeOptions>(m, "DecodeOptions")
      .def(py::init<>())
      .def_readwrite("skip_special_tokens", &DecodeOptions::skip_special_tokens);

  py::class_<TrainerOptions>(m, "TrainerOptions")
      .def(py::init<>())
      .def_readwrite("vocab_size", &TrainerOptions::vocab_size)
      .def_readwrite("min_frequency", &TrainerOptions::min_frequency)
      .def_readwrite("top_k_pairs", &TrainerOptions::top_k_pairs)
      .def_readwrite("chunk_size", &TrainerOptions::chunk_size)
      .def_readwrite("max_memory_bytes", &TrainerOptions::max_memory_bytes)
      .def_readwrite("num_threads", &TrainerOptions::num_threads)
      .def_readwrite("byte_fallback", &TrainerOptions::byte_fallback);

  py::class_<TrainResult>(m, "TrainResult")
      .def(py::init<>())
      .def_readwrite("vocab", &TrainResult::vocab)
      .def_readwrite("merges", &TrainResult::merges);

  py::class_<ByteLevelBPETrainer>(m, "ByteLevelBPETrainer")
      .def(py::init<TrainerOptions>())
      .def("train_from_files", &ByteLevelBPETrainer::TrainFromFiles)
      .def("train_from_lines", &ByteLevelBPETrainer::TrainFromLines);

  py::class_<ByteLevelBPETokenizer>(m, "ByteLevelBPETokenizer")
      .def(py::init<Vocab, MergeRules, TokenId, TokenId, TokenId>(), py::arg("vocab"), py::arg("merges"),
           py::arg("unk_id"), py::arg("bos_id") = static_cast<TokenId>(-1),
           py::arg("eos_id") = static_cast<TokenId>(-1))
      .def_static("from_files", &ByteLevelBPETokenizer::FromFiles)
      .def("encode", &ByteLevelBPETokenizer::Encode, py::arg("text"), py::arg("opts") = EncodeOptions{})
      .def("decode", [](const ByteLevelBPETokenizer& self, const std::vector<TokenId>& ids,
                        DecodeOptions opts) { return self.Decode(ids, opts); })
      .def("vocab_size", &ByteLevelBPETokenizer::VocabSize)
      .def("token_by_id", &ByteLevelBPETokenizer::TokenById)
      .def("id_by_token", &ByteLevelBPETokenizer::IdByToken)
      .def_property_readonly("vocab", &ByteLevelBPETokenizer::GetVocab)
      .def_property_readonly("merges", &ByteLevelBPETokenizer::GetMerges);

  py::class_<PretokenizeOptions>(m, "PretokenizeOptions")
      .def(py::init<>())
      .def_readwrite("chunk_bytes", &PretokenizeOptions::chunk_bytes)
      .def_readwrite("num_threads", &PretokenizeOptions::num_threads)
      .def_readwrite("mmap_output", &PretokenizeOptions::mmap_output);

  py::class_<PretokenizeEngine>(m, "PretokenizeEngine")
      .def(py::init([](const ByteLevelBPETokenizer& tk) { return PretokenizeEngine(tk); }),
           py::keep_alive<1, 2>())
      .def("pretokenize_files", &PretokenizeEngine::PretokenizeFiles);

  py::class_<StreamingEncoder>(m, "StreamingEncoder")
      .def(py::init([](const ByteLevelBPETokenizer& tk, std::size_t flush_chars) {
             return StreamingEncoder(tk, flush_chars);
           }),
           py::keep_alive<1, 2>(), py::arg("tokenizer"), py::arg("flush_chars") = 4096)
      .def("push", &StreamingEncoder::Push)
      .def("flush", &StreamingEncoder::Flush);

  m.def("save_as_gpt2", &SaveAsGPT2);
  m.def("save_as_hf_tokenizer_json", &SaveAsHFTokenizerJson, py::arg("result"), py::arg("tokenizer_json_path"),
        py::arg("model_type") = "BPE");
}

