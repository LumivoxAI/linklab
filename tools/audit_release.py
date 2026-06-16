from __future__ import annotations

import tarfile
import zipfile
import argparse
import tempfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
DIST = ROOT / "dist"
PACKAGE = ROOT / "src" / "lumivox_linklab"
SDIST_PUBLIC_TREES = ("doc", "examples", "fixtures")
SDIST_ROOT_FILES = {
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "justfile",
    "pyproject.toml",
    "pyproject.toml.orig",
    "tools/audit_release.py",
    "tools/generate_golden_fixtures.py",
    "uv.lock",
}


def _files_below(directory: Path, prefix: str = "") -> set[str]:
    return {
        f"{prefix}{path.relative_to(directory).as_posix()}"
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    }


def _single(pattern: str) -> Path:
    matches = tuple(DIST.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"expected one {pattern} artifact, found {matches}")
    return matches[0]


def _audit_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}

    package_files = _files_below(PACKAGE, "lumivox_linklab/")
    metadata = {name for name in names if ".dist-info/" in name}
    suffixes = {name.split(".dist-info/", 1)[1] for name in metadata}
    expected_suffixes = {"METADATA", "RECORD", "WHEEL", "licenses/LICENSE"}
    if names - metadata != package_files or suffixes != expected_suffixes:
        raise AssertionError(
            f"wheel manifest mismatch: unexpected={sorted((names - metadata) - package_files)}, "
            f"missing={sorted(package_files - (names - metadata))}, metadata={sorted(suffixes)}"
        )


def _audit_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in members}
    if len(roots) != 1:
        raise AssertionError(f"sdist has unexpected roots: {sorted(roots)}")
    root = roots.pop()
    names = {name.removeprefix(f"{root}/") for name in members}

    expected = set(SDIST_ROOT_FILES)
    expected.update(_files_below(PACKAGE, "src/lumivox_linklab/"))
    for tree in SDIST_PUBLIC_TREES:
        expected.update(_files_below(ROOT / tree, f"{tree}/"))
    if names != expected:
        raise AssertionError(
            f"sdist manifest mismatch: unexpected={sorted(names - expected)}, missing={sorted(expected - names)}"
        )


def _audit_install(wheel: Path, python: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"linklab-{python}-") as temporary:
        environment = Path(temporary) / ".venv"
        subprocess.run(["uv", "venv", "--python", python, str(environment)], check=True)
        interpreter = environment / "bin" / "python"
        subprocess.run(["uv", "pip", "install", "--python", str(interpreter), str(wheel)], check=True)
        subprocess.run(
            [
                str(interpreter),
                "-c",
                (
                    "from importlib.metadata import version; from importlib.resources import files; "
                    "import lumivox_linklab; "
                    "assert version('lumivox-linklab') == '0.1.0'; "
                    "assert files(lumivox_linklab).joinpath('py.typed').is_file()"
                ),
            ],
            check=True,
            cwd=temporary,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit built Linklab release artifacts")
    parser.add_argument("--python", action="append", default=[], help="Python version for an isolated install check")
    arguments = parser.parse_args()

    wheel = _single("*.whl")
    sdist = _single("*.tar.gz")
    _audit_wheel(wheel)
    _audit_sdist(sdist)
    for python in arguments.python:
        _audit_install(wheel, python)


if __name__ == "__main__":
    main()
