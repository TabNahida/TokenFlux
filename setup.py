from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
EXTENSION_SUFFIXES = {".pyd", ".so", ".dylib"}


def _find_built_extension() -> Path:
    candidates: list[Path] = []
    build_root = ROOT / "build"
    if not build_root.exists():
        raise FileNotFoundError("xmake build directory was not created")

    for path in build_root.rglob("tokenflux_cpp*"):
        if not path.is_file():
            continue
        if not path.name.startswith("tokenflux_cpp"):
            continue
        if path.suffix.lower() not in EXTENSION_SUFFIXES:
            continue
        if ".objs" in path.parts or ".deps" in path.parts:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError("could not find tokenflux_cpp extension artifact under build/")

    return max(candidates, key=lambda item: item.stat().st_mtime)


class XMakeBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        xmake = os.environ.get("XMAKE_BINARY") or shutil.which("xmake")
        if not xmake:
            raise RuntimeError("xmake executable was not found in PATH; install xmake before running pip install")

        env = os.environ.copy()
        env["PYTHON_EXECUTABLE"] = sys.executable
        if "XMAKE_GLOBALDIR" in env:
            xmake_globaldir = Path(env["XMAKE_GLOBALDIR"])
            xmake_globaldir.mkdir(parents=True, exist_ok=True)
        else:
            build_temp = Path(self.build_temp)
            build_temp.mkdir(parents=True, exist_ok=True)
            xmake_globaldir = Path(tempfile.mkdtemp(prefix="xmake-global-", dir=build_temp))
        env["XMAKE_GLOBALDIR"] = str(xmake_globaldir)

        mode = "debug" if self.debug else "release"
        subprocess.run([xmake, "f", "-y", "-m", mode, "--pybind=y"], cwd=ROOT, env=env, check=True)
        subprocess.run([xmake, "build", "-y", "tokenflux_cpp"], cwd=ROOT, env=env, check=True)

        built_extension = _find_built_extension()
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_extension, destination)


setup(
    ext_modules=[Extension("tokenflux_cpp", sources=[])],
    cmdclass={"build_ext": XMakeBuildExt},
)
