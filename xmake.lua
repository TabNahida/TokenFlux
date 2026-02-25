set_project("TokenFlux")
set_version("0.1.0")

add_rules("mode.debug", "mode.release")
set_languages("c++23")

add_requires("zlib", "xz")

target("byte_bpe_train")
    set_kind("binary")
    add_files(
        "tokenizer/byte_bpe_train.cpp",
        "tokenizer/byte_bpe_bpe.cpp",
        "tokenizer/byte_bpe_lib.cpp"
    )
    set_rundir("$(projectdir)")
    add_packages("zlib", "xz")

target("prepare_shards")
    set_kind("binary")
    add_files(
        "tokenizer/prepare_shards.cpp",
        "tokenizer/byte_bpe_lib.cpp"
    )
    set_rundir("$(projectdir)")
    add_packages("zlib", "xz")
