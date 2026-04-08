# Repository development workflows.

default:
    @just --list

# Create or synchronize the complete local development environment.
postclone:
    uv venv --python /usr/bin/python3 --system-site-packages
    uv sync --python /usr/bin/python3 --all-groups --all-extras

# Synchronize the default project and development dependencies.
sync:
    uv sync

# Resolve and update the committed dependency lockfile.
lock:
    uv lock

# Verify that the lockfile matches pyproject.toml.
lock_check:
    uv lock --check

fmt:
    uv run ruff format .

fmt_check:
    uv run ruff format --check .

lint:
    uv run ruff check .

lint_fix:
    uv run ruff check --fix .

typecheck:
    uv run mypy

test:
    uv run pytest

# Test with every dependency group and optional feature enabled.
test_all:
    uv run --all-groups --all-extras pytest

# Regenerate the reviewable v1 MessagePack golden manifest explicitly.
fixtures:
    uv run python tools/generate_golden_fixtures.py

# Verify that committed golden bytes match the canonical encoder.
fixtures_check:
    uv run python tools/generate_golden_fixtures.py --check

# Build artifacts without local uv source overrides.
build:
    uv build --no-sources --clear

# Run the fast, required checks before creating a commit.
precommit: fmt_check lint typecheck test fixtures_check lock_check
