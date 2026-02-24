set_project("TokenFlux")
set_version("0.1.0")
set_languages("cxx20")

add_rules("mode.debug", "mode.release")

option("python")
    set_default(false)
    set_showmenu(true)
option_end()

option("parquet")
    set_default(false)
    set_showmenu(true)
option_end()

add_requires("nlohmann_json")
add_requires("zlib")

if has_config("python") then
    add_requires("pybind11")
end

if has_config("parquet") then
    add_requires("arrow")
end

target("tokenflux")
    set_kind("static")
    add_headerfiles("include/(tokenflux/**.hpp)")
    add_includedirs("include", {public = true})
    add_packages("nlohmann_json", "zlib", {public = true})
    if has_config("parquet") then
        add_packages("arrow", {public = true})
        add_defines("TOKENFLUX_WITH_PARQUET")
    end
    add_files("src/**.cpp")

target("tokenflux_cli")
    set_kind("binary")
    add_files("cli/main.cpp")
    add_deps("tokenflux")

if has_config("python") then
    target("py_tokenflux")
        set_kind("shared")
        set_basename("tokenflux")
        add_rules("python.library")
        add_files("python/bindings.cpp")
        add_deps("tokenflux")
        add_packages("pybind11")
end
