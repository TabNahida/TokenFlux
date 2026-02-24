set_project("TokenFlux")
set_version("0.1.0")
set_languages("c++20")
add_rules("mode.debug", "mode.release")

add_requires("nlohmann_json", "pybind11")

option("python")
    set_default(false)
    set_showmenu(true)
    set_description("build python module")
option_end()

target("tokenflux")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_headerfiles("include/(tokenflux/*.hpp)")
    add_files("src/*.cpp")
    add_packages("nlohmann_json")
    if is_plat("linux") then
        add_syslinks("pthread")
    end

target("tokenflux_cli")
    set_kind("binary")
    add_files("examples/main.cpp")
    add_deps("tokenflux")
    add_includedirs("include")

if has_config("python") then
    target("pytokenflux")
        set_kind("shared")
        set_basename("pytokenflux")
        add_files("python/bindings.cpp")
        add_includedirs("include")
        add_deps("tokenflux")
        add_packages("pybind11")
        if is_plat("linux") then
            add_ldflags("-Wl,-undefined,dynamic_lookup", {force = true})
        end
end


target("smoke_test")
    set_kind("binary")
    add_files("tests/smoke.cpp")
    add_deps("tokenflux")
    add_includedirs("include")
