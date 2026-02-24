#include "tokenflux/tokenizer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(tokenflux, m) {
    py::class_<tokenflux::Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("load_hf_tokenizer_json", &tokenflux::Tokenizer::load_hf_tokenizer_json)
        .def("save_hf_tokenizer_json", &tokenflux::Tokenizer::save_hf_tokenizer_json)
        .def("tokenize", [](const tokenflux::Tokenizer& t, const std::string& s) { return t.tokenize(s); })
        .def("detokenize", &tokenflux::Tokenizer::detokenize);
}
