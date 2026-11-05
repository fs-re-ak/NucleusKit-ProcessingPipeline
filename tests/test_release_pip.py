"""tools/release_pip.py: parsing, resumable step planning and the order of the publishing commands (no network)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tomllib")

_PATH = Path(__file__).resolve().parents[1] / "tools" / "release_pip.py"
_spec = importlib.util.spec_from_file_location("release_pip", _PATH)
rp = importlib.util.module_from_spec(_spec)
sys.modules["release_pip"] = rp
_spec.loader.exec_module(rp)

CHANGELOG = """# Changelog

## 1.2.0 (pre-release note)

### Fixed

- Something.

## 1.1.0

## 1.0.0

- First.
"""


def test_changelog_section() -> None:
    assert rp.changelog_section(CHANGELOG, "1.2.0") == "### Fixed\n\n- Something."
    assert rp.changelog_section(CHANGELOG, "1.0.0") == "- First."
    assert rp.changelog_section(CHANGELOG, "1.1.0") is None  # empty
    assert rp.changelog_section(CHANGELOG, "1.2") is None  # a prefix is not the version
    assert rp.changelog_section(CHANGELOG, "9.9.9") is None


def test_init_version() -> None:
    assert rp.init_version('"""Doc."""\n\n__version__ = "1.0.2"\n') == "1.0.2"
    assert rp.init_version("x = 1\n") is None


def test_is_newer() -> None:
    assert rp.is_newer("1.0.10", {"1.0.9", "1.0.2", "1.0.0a1"})
    assert not rp.is_newer("1.0.2", {"1.0.2"})
    assert not rp.is_newer("1.0.1", {"1.0.2"})
    assert not rp.is_newer("1.0.0a1", set())
    assert rp.is_newer("1.0.0", set())


def test_dependencies() -> None:
    deps = ("numpy>=2.5,<3", "re-ak-core>=1.0.2,<1.1", 'pyqtgraph==0.14 ; extra == "gui"')
    assert rp.internal_floors(deps) == {"re-ak-core": "1.0.2"}
    assert rp.direct_url_dependencies(deps) == []
    url = "re-ak-core @ git+https://github.com/fs-re-ak/re-ak-core.git@1.0.0a1"
    assert rp.direct_url_dependencies((*deps, url)) == [url]


@pytest.mark.parametrize(
    ("tag_pushed", "on_testpypi", "released", "expected"),
    [
        (False, False, False, ["tests", "tag", "wait_testpypi", "verify", "confirm", "release", "wait_pypi"]),
        (True, False, False, ["wait_testpypi", "verify", "confirm", "release", "wait_pypi"]),
        (True, True, False, ["verify", "confirm", "release", "wait_pypi"]),
        (True, True, True, ["wait_pypi"]),
        (True, False, True, ["wait_pypi"]),  # a release exists: never go back to TestPyPI
    ],
)
def test_plan_steps_resume(tag_pushed: bool, on_testpypi: bool, released: bool, expected: list[str]) -> None:
    assert rp.plan_steps(rp.State(tag_pushed, on_testpypi, released)) == expected


class FakeWorld:
    """Stands in for git, gh and the package indexes; records the commands that change something."""

    HEAD = "a" * 40

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str = "1.0.3") -> None:
        self.root = tmp_path
        self.version = version
        self.tag_commit: str | None = None
        self.released = False
        self.branch = "main"
        self.dirty = ""
        self.testpypi: set[str] = set()
        self.pypi = {"1.0.2"}
        self.required: set[str] = {"1.0.2"}
        self.actions: list[str] = []
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "re-ak-core"\nversion = "{version}"\ndependencies = []\n', encoding="utf-8")
        package = tmp_path / "src" / "reak_core"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
        (tmp_path / "CHANGELOG.md").write_text(f"# Changelog\n\n## {version}\n\n- Fix.\n", encoding="utf-8")
        workflows = tmp_path / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "publish.yml").write_text("name: Publish\n", encoding="utf-8")
        monkeypatch.setattr(rp, "sh", self.sh)
        monkeypatch.setattr(rp, "pypi_versions", self.versions)
        monkeypatch.setattr(rp, "run_tests", lambda project: self.actions.append("tests"))
        monkeypatch.setattr(rp, "verify_testpypi_package", lambda project: self.actions.append("verify"))
        monkeypatch.setattr(rp.time, "sleep", lambda s: None)
        monkeypatch.setattr(rp.shutil, "which", lambda tool: tool)
        monkeypatch.setattr("builtins.input", lambda prompt="": self.answer(prompt))
        self.answers = ["yes"]

    def answer(self, prompt: str) -> str:
        self.actions.append("prompt")
        return self.answers.pop(0)

    def versions(self, host: str, name: str) -> set[str]:
        return self.testpypi if host == rp.TEST_HOST else self.pypi

    def sh(self, *cmd: str, check: bool = True, **_: object) -> subprocess.CompletedProcess[str]:
        out, code = "", 0
        match cmd[:3]:
            case ("gh", "release", "view"):
                code = 0 if self.released else 1
            case ("gh", "release", "create"):
                self.actions.append("release")
                self.released = True
                self.pypi.add(self.version)
            case ("gh", "run", "list"):
                out = json.dumps([{"databaseId": 1, "status": "completed", "conclusion": "success", "url": "u"}])
            case ("git", "ls-remote", _):
                out = f"{self.tag_commit}\trefs/tags/{self.version}\n" if self.tag_commit else ""
            case ("git", "status", _):
                out = self.dirty
            case ("git", "rev-parse", _):
                out = self.HEAD
            case ("git", "branch", _):
                out = self.branch
            case ("git", "tag", _):
                self.actions.append("tag")
            case ("git", "push", _):
                self.actions.append("push")
                self.tag_commit = self.HEAD
                self.testpypi.add(self.version)
        return subprocess.CompletedProcess(cmd, code, out, "")


def test_full_release_runs_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = FakeWorld(tmp_path, monkeypatch)
    assert rp.release(tmp_path) == 0
    assert world.actions == ["tests", "tag", "push", "verify", "prompt", "release"]
    assert "1.0.3" in world.pypi


def test_answering_no_stops_before_pypi_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = FakeWorld(tmp_path, monkeypatch)
    world.answers = ["no", "yes"]
    with pytest.raises(rp.ReleaseError):
        rp.release(tmp_path)
    assert not world.released and "1.0.3" not in world.pypi

    world.actions.clear()
    assert rp.release(tmp_path) == 0
    assert world.actions == ["verify", "prompt", "release"]  # neither tests nor tag again


def test_version_already_on_pypi_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = FakeWorld(tmp_path, monkeypatch, version="1.0.2")
    assert rp.release(tmp_path) == 0
    assert world.actions == []


def test_dry_run_publishes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    world = FakeWorld(tmp_path, monkeypatch)
    assert rp.release(tmp_path, dry_run=True) == 0
    assert world.actions == []
    assert "tests, tag, wait_testpypi" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda w: setattr(w, "branch", "feature"), "Run this from main"),
        (lambda w: setattr(w, "dirty", " M file.py"), "uncommitted changes"),
        (lambda w: w.testpypi.add("1.0.3"), "cannot be uploaded twice"),
        (lambda w: setattr(w, "tag_commit", "b" * 40), "git switch --detach 1.0.3"),
        (lambda w: (w.root / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8"), "CHANGELOG.md"),
        (lambda w: (w.root / "src/reak_core/__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8"),
         "__init__.py says version 1.0.0"),
    ],
)
def test_failed_checks_publish_nothing(tmp_path: Path, monkeypatch, capsys, change, message) -> None:
    world = FakeWorld(tmp_path, monkeypatch)
    change(world)
    assert rp.release(tmp_path) == 1
    assert world.actions == []
    assert message in capsys.readouterr().out


def test_failed_run_is_not_treated_as_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = FakeWorld(tmp_path, monkeypatch)
    failed = json.dumps([{"databaseId": 1, "status": "completed", "conclusion": "failure", "url": "u"}])
    original = world.sh

    def sh(*cmd: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ("gh", "run", "list"):
            return subprocess.CompletedProcess(cmd, 0, failed, "")
        return original(*cmd, **kwargs)

    monkeypatch.setattr(rp, "sh", sh)
    world.answers = ["n"]
    with pytest.raises(rp.ReleaseError, match="run this script again"):
        rp.wait_for_run("1.0.3", "push", "TestPyPI")
