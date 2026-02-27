set_project("TokenFlux")
set_version("0.3.0")

add_rules("mode.debug", "mode.release")
set_languages("c++23")

add_requires("zlib", "xz")

local function detect_python_binding()
    local localappdata = os.getenv("LOCALAPPDATA")
    if not localappdata then
        return nil
    end
    local python_roots = os.dirs(path.join(localappdata, "Programs", "Python", "Python*"))
    if not python_roots or #python_roots == 0 then
        return nil
    end
    table.sort(python_roots)
    local python_root = python_roots[#python_roots]
    local version_text = path.basename(python_root):match("^Python(%d+)$")
    if not version_text then
        return nil
    end

    local pybind11_include = os.getenv("PYBIND11_INCLUDE_DIR")
    if not pybind11_include or pybind11_include == "" then
        local userprofile = os.getenv("USERPROFILE")
        if userprofile then
            local torch_includes = os.dirs(path.join(userprofile, ".conda", "envs", "*", "Lib", "site-packages", "torch", "include"))
            if torch_includes and #torch_includes > 0 then
                table.sort(torch_includes)
                pybind11_include = torch_includes[#torch_includes]
            end
        end
    end

    return {
        include_dir = path.join(python_root, "Include"),
        lib_dir = path.join(python_root, "libs"),
        lib_name = "python" .. version_text,
        pybind11_include = pybind11_include or ""
    }
end

local python_binding_config = nil

option("python_binding")
    set_default(false)
    set_showmenu(true)
    set_description("Build Python bindings with pybind11")
    on_check(function (option)
        if not option:enabled() then
            python_binding_config = nil
            return
        end

        local python = detect_python_binding()
        if not python then
            raise("failed to detect Python build settings for tokenflux_cpp")
        end
        if python.include_dir == "" or python.lib_dir == "" or python.lib_name == "" then
            raise("incomplete Python build settings for tokenflux_cpp")
        end
        if python.pybind11_include == "" then
            raise("pybind11 headers were not found; set PYBIND11_INCLUDE_DIR if needed")
        end
        python_binding_config = python
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
    add_packages("zlib", "xz")
    if is_plat("windows") then
        add_syslinks("winhttp")
    end

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
    add_packages("zlib", "xz")
    if is_plat("windows") then
        add_syslinks("winhttp")
    end

target("tokenflux_cpp")
    set_kind("shared")
    set_basename("tokenflux_cpp")
    set_prefixname("")
    set_extension(".pyd")
    add_options("python_binding")
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
    add_packages("zlib", "xz")
    if is_plat("windows") then
        add_syslinks("winhttp")
    end
    on_load(function (target)
        if not get_config("python_binding") then
            target:set("enabled", false)
            return
        end

        target:set("enabled", true)
        local python = python_binding_config or detect_python_binding()
        target:add("includedirs", python.include_dir, python.pybind11_include)
        target:add("linkdirs", python.lib_dir)
        target:add("links", python.lib_name)
    end)
