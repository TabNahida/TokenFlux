set_project("TokenFlux")
set_version("0.3.2")

add_rules("mode.debug", "mode.release")
set_languages("c++23")

add_requires("zlib", "xz")
add_requires("cpp-httplib", {configs = {ssl = true}})
add_requires("python 3.x", {alias = "python", optional = true})
add_requires("pybind11", {optional = true})

option("pybind")
    set_default(false)
    set_showmenu(true)
    set_description("Build Python bindings with pybind11")
    on_check(function (option)
        if not option:enabled() then
            return
        end
        if not has_package("python") or not has_package("pybind11") then
            raise("option `pybind` requires packages `python 3.x` and `pybind11`")
        end
    end)
option_end()

target("TokenFluxTrain")
    set_kind("binary")
    add_files(
        "tokenizer/TokenFluxTrain.cpp",
        "tokenizer/input_source.cpp",
        "tokenizer/tokenflux_lib.cpp",
        "tokenizer/train_frontend.cpp",
        "tokenizer/train_io.cpp",
        "tokenizer/train_pipeline.cpp",
        "tokenizer/trainers.cpp",
        "tokenizer/train_backend_common.cpp",
        "tokenizer/train_backend_byte_bpe.cpp",
        "tokenizer/train_backend_bpe.cpp",
        "tokenizer/train_backend_wordpiece.cpp",
        "tokenizer/train_backend_unigram.cpp"
    )
    set_rundir("$(projectdir)")
    add_packages("zlib", "xz", "cpp-httplib")

target("TokenFluxTokenize")
    set_kind("binary")
    add_files(
        "tokenizer/TokenFluxTokenize.cpp",
        "tokenizer/input_source.cpp",
        "tokenizer/tokenize_common.cpp",
        "tokenizer/tokenize_tokenizer.cpp",
        "tokenizer/tokenize_pipeline.cpp",
        "tokenizer/tokenflux_lib.cpp"
    )
    set_rundir("$(projectdir)")
    add_packages("zlib", "xz", "cpp-httplib")

target("tokenflux_cpp")
    set_kind("shared")
    add_rules("python.module")
    set_basename("tokenflux_cpp")
    add_options("pybind")
    add_files(
        "tokenizer/input_source.cpp",
        "tokenizer/tokenflux_lib.cpp",
        "tokenizer/train_frontend.cpp",
        "tokenizer/train_io.cpp",
        "tokenizer/train_pipeline.cpp",
        "tokenizer/trainers.cpp",
        "tokenizer/train_backend_common.cpp",
        "tokenizer/train_backend_byte_bpe.cpp",
        "tokenizer/train_backend_bpe.cpp",
        "tokenizer/train_backend_wordpiece.cpp",
        "tokenizer/train_backend_unigram.cpp",
        "tokenizer/tokenize_common.cpp",
        "tokenizer/tokenize_tokenizer.cpp",
        "tokenizer/tokenize_pipeline.cpp",
        "tokenizer/tokenflux_pybind.cpp"
    )
    add_packages("zlib", "xz", "cpp-httplib")
    on_load(function (target)
        if not get_config("pybind") then
            target:set("enabled", false)
            return
        end

        target:set("enabled", true)
        target:add("packages", "python", "pybind11")
    end)
