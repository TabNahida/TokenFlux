set_project("TokenFlux")
set_version("0.2.3")

add_rules("mode.debug", "mode.release")
set_languages("c++23")

add_requires("zlib", "xz")

target("TokenFluxTrain")
    set_kind("binary")
    add_files(
        "tokenizer/TokenFluxTrain.cpp",
        "tokenizer/tokenflux_bpe.cpp",
        "tokenizer/tokenflux_lib.cpp",
        "tokenizer/train_frontend.cpp",
        "tokenizer/train_io.cpp",
        "tokenizer/trainers.cpp",
        "tokenizer/train_backend_common.cpp",
        "tokenizer/train_backend_byte_bpe.cpp",
        "tokenizer/train_backend_bpe.cpp",
        "tokenizer/train_backend_wordpiece.cpp",
        "tokenizer/train_backend_unigram.cpp"
    )
    set_rundir("$(projectdir)")
    add_packages("zlib", "xz")

target("TokenFluxTokenize")
    set_kind("binary")
    add_files(
        "tokenizer/TokenFluxTokenize.cpp",
        "tokenizer/tokenflux_lib.cpp"
    )
    set_rundir("$(projectdir)")
    add_packages("zlib", "xz")
