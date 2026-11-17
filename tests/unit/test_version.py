"""The release workflow refuses a tag that disagrees with these; catch drift earlier."""

import os
import re
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pyproject_version():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _package_version():
    with open(os.path.join(ROOT, "app", "__init__.py"), encoding="utf-8") as f:
        return re.search(r'^__version__ = "([^"]+)"', f.read(), re.M).group(1)


def test_package_version_matches_pyproject():
    assert _package_version() == _pyproject_version()


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _pyproject_version())


def test_changelog_has_entry_for_version():
    with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as f:
        assert f"## [{_pyproject_version()}]" in f.read()
