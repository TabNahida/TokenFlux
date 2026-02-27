from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
EXTENSION_NAMES = {"tokenflux_cpp.pyd", "tokenflux_cpp.so"}


def _resolve_pybind11_include() -> str | None:
    include_dir = os.environ.get("PYBIND11_INCLUDE_DIR")
    if include_dir:
        return include_dir

    try:
        import pybind11
    except Exception:
        pybind11 = None

    if pybind11 is not None:
        return pybind11.get_include()

    try:
        import torch
    except Exception:
        return None

    candidate = Path(torch.__file__).resolve().parent / "include"
    if (candidate / "pybind11" / "pybind11.h").exists():
        return str(candidate)
    return None


def _find_built_extension() -> Path:
    candidates: list[Path] = []
    build_root = ROOT / "build"
    if not build_root.exists():
        raise FileNotFoundError("xmake build directory was not created")

    for path in build_root.rglob("tokenflux_cpp.*"):
        if path.name not in EXTENSION_NAMES:
            continue
        if ".objs" in path.parts or ".deps" in path.parts:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError("could not find tokenflux_cpp extension artifact under build/")

    return max(candidates, key=lambda item: item.stat().st_mtime)


def _normalize_python_lib_name(name: str) -> str:
    normalized = name
    for suffix in (".lib", ".dll", ".so", ".dylib"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    if normalized.startswith("lib"):
        normalized = normalized[3:]
    return normalized


class XMakeBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        xmake = os.environ.get("XMAKE_BINARY") or shutil.which("xmake")
        if not xmake:
            raise RuntimeError("xmake executable was not found in PATH; install xmake before running pip install")

        pybind11_include = _resolve_pybind11_include()
        if not pybind11_include:
            raise RuntimeError(
                "pybind11 headers were not found; install pybind11 or set PYBIND11_INCLUDE_DIR before building"
            )

        env = os.environ.copy()
        env["PYTHON_EXECUTABLE"] = sys.executable
        env["PYBIND11_INCLUDE_DIR"] = pybind11_include
        python_include_dir = sysconfig.get_path("include") or ""
        python_lib_dir = sysconfig.get_config_var("LIBDIR") or str(Path(sys.base_prefix) / "libs")
        python_lib_name = _normalize_python_lib_name(
            sysconfig.get_config_var("LDLIBRARY")
            or sysconfig.get_config_var("LIBRARY")
            or f"python{sys.version_info.major}{sys.version_info.minor}"
        )
        env["PYTHON_INCLUDE_DIR"] = python_include_dir
        env["PYTHON_LIB_DIR"] = python_lib_dir
        env["PYTHON_LIB_NAME"] = python_lib_name
        if "XMAKE_GLOBALDIR" in env:
            xmake_globaldir = Path(env["XMAKE_GLOBALDIR"])
            xmake_globaldir.mkdir(parents=True, exist_ok=True)
        else:
            build_temp = Path(self.build_temp)
            build_temp.mkdir(parents=True, exist_ok=True)
            xmake_globaldir = Path(tempfile.mkdtemp(prefix="xmake-global-", dir=build_temp))
        env["XMAKE_GLOBALDIR"] = str(xmake_globaldir)

        mode = "debug" if self.debug else "release"
        subprocess.run([xmake, "f", "-y", "-m", mode, "--python_binding=y"], cwd=ROOT, env=env, check=True)
        subprocess.run([xmake, "build", "-y", "tokenflux_cpp"], cwd=ROOT, env=env, check=True)

        built_extension = _find_built_extension()
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_extension, destination)


setup(
    ext_modules=[Extension("tokenflux_cpp", sources=[])],
    cmdclass={"build_ext": XMakeBuildExt},
)
