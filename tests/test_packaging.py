import re
import tomllib
from pathlib import Path
from importlib.resources import files

import lumivox_linklab

ROOT = Path(__file__).parents[1]
LOCKED_CORE_PATTERN = re.compile(
    r'source = \{ git = "https://github\.com/LumivoxAI/core\.git\?rev=master#[0-9a-f]{40}" \}'
)


def _project_metadata() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as pyproject:
        document = tomllib.load(pyproject)
    return document["project"]  # type: ignore[no-any-return]


def test_distribution_metadata() -> None:
    project = _project_metadata()

    assert project["name"] == "lumivox-linklab"
    assert project["requires-python"] == ">=3.13,<3.15"
    assert project["license"] == "Apache-2.0"
    assert project["dependencies"] == [
        "lumivox-core @ git+https://github.com/LumivoxAI/core.git@master",
        "msgpack>=1.1,<2",
        "websockets>=15,<17",
        "zeroconf>=0.151,<0.152",
    ]


def test_core_dependency_tracks_master_with_locked_resolution() -> None:
    dependencies = _project_metadata()["dependencies"]

    assert isinstance(dependencies, list)
    core_dependencies = [dependency for dependency in dependencies if str(dependency).startswith("lumivox-core @ ")]
    assert core_dependencies == ["lumivox-core @ git+https://github.com/LumivoxAI/core.git@master"]
    assert LOCKED_CORE_PATTERN.search((ROOT / "uv.lock").read_text())


def test_package_is_importable_and_typed() -> None:
    package_files = files(lumivox_linklab)

    assert package_files.joinpath("py.typed").is_file()
