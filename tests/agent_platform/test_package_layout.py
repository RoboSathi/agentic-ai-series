"""Every module lives in a subpackage named for what it is.

A file at a package root (`money.py`, `narration.py`) says nothing about
where it belongs, and the root becomes the place things go when nobody
decided. Only the package's own `__init__.py` and `__main__.py` may sit
there.
"""

import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
ALLOWED_AT_ROOT = frozenset({"__init__.py", "__main__.py"})


def _packages() -> list[pathlib.Path]:
    return sorted(path.parent for path in SRC.glob("*/__init__.py"))


@pytest.mark.parametrize("package", _packages(), ids=lambda path: path.name)
def test_no_module_sits_at_a_package_root(package: pathlib.Path) -> None:
    loose = sorted(path.name for path in package.glob("*.py") if path.name not in ALLOWED_AT_ROOT)
    assert loose == [], (
        f"{package.name} has modules at its root: {', '.join(loose)}. "
        "Move each into the subpackage that names what it is."
    )
