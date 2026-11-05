# SPDX-License-Identifier: GPL-3.0-or-later
"""Release this repository's package: TestPyPI first, then PyPI, from what is already on ``main``.

Run it through ``deploy_pip_package.bat`` at the repository root. The same file lives in re-ak-core and in
NucleusKit-ProcessingPipeline; it reads the package name and version from ``pyproject.toml``.

The upload itself is done by ``.github/workflows/publish.yml`` (Trusted Publishing, no token): a pushed tag uploads
to TestPyPI, a published GitHub release uploads to PyPI. This script checks everything first, drives those two
triggers, waits for the runs, tests the TestPyPI package in a throwaway venv and asks for confirmation before PyPI.

A version can only be uploaded once, so the script is resumable: run it again after any failure or Ctrl+C and it
continues from what is already done. Version bumps and merges stay in pull requests.

Options: ``--dry-run`` stops after the checks and prints the remaining steps; ``--skip-tests`` skips ruff and pytest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    sys.exit("This script needs Python 3.11 or newer.")

WORKFLOW = "publish.yml"
DEFAULT_BRANCH = "main"
TEST_HOST = "test.pypi.org"
PROD_HOST = "pypi.org"
INTERNAL_PACKAGES = ("re-ak-core",)  # must already be on PyPI when a package depends on them
# Import package and a module that pulls in the processing code, for the check of the installed package.
IMPORTS = {
    "re-ak-core": ("reak_core", "reak_core.pipeline"),
    "nucleuskit-toolkit": ("nucleuskit_toolkit", "nucleuskit_toolkit.pipeline"),
}
POLL_SECONDS = 10
RUN_START_TIMEOUT_S = 180
INDEX_TIMEOUT_S = 600
INSTALL_ATTEMPTS = 6
INSTALL_RETRY_S = 30


class ReleaseError(Exception):
    """A step failed; the message says what to do."""


@dataclass(frozen=True)
class Project:
    root: Path
    name: str
    version: str
    dependencies: tuple[str, ...]

    @property
    def import_name(self) -> str:
        return IMPORTS[self.name][0]

    @property
    def smoke_module(self) -> str:
        return IMPORTS[self.name][1]

    @property
    def source_root(self) -> Path:
        """Folder holding the import package (``src`` layout or repository root)."""
        src = self.root / "src"
        return src if (src / self.import_name).is_dir() else self.root


@dataclass(frozen=True)
class State:
    tag_pushed: bool
    on_testpypi: bool
    released: bool


# ── Pure helpers ──────────────────────────────────────────────────────────────


def read_project(root: Path) -> Project:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    name = data["name"]
    if name not in IMPORTS:
        raise ReleaseError(f"Unknown package {name!r}: add it to IMPORTS in {Path(__file__).name}.")
    return Project(root, name, data["version"], tuple(data.get("dependencies", ())))


def init_version(text: str) -> str | None:
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else None


def changelog_section(text: str, version: str) -> str | None:
    """Body of the ``## <version>`` section (the heading may carry a suffix), or None when missing or empty."""
    match = re.search(rf"^## {re.escape(version)}(?![\w.]).*$", text, re.MULTILINE)
    if match is None:
        return None
    rest = text[match.end():]
    following = re.search(r"^## ", rest, re.MULTILINE)
    body = (rest[: following.start()] if following else rest).strip()
    return body or None


def version_key(version: str) -> tuple[int, ...] | None:
    """Numeric key of a final ``x.y.z`` version; None for anything else (pre-releases)."""
    return tuple(int(p) for p in version.split(".")) if re.fullmatch(r"\d+\.\d+\.\d+", version) else None


def is_newer(version: str, published: set[str]) -> bool:
    key = version_key(version)
    keys = [k for k in map(version_key, published) if k is not None]
    return key is not None and all(key > k for k in keys)


def internal_floors(dependencies: tuple[str, ...]) -> dict[str, str]:
    """Lowest accepted version of each internal package among the dependencies."""
    floors = {}
    for dep in dependencies:
        match = re.match(r"([A-Za-z0-9._-]+)[^;]*?(?:>=|==)\s*([\w.]+)", dep)
        if match and match.group(1) in INTERNAL_PACKAGES:
            floors[match.group(1)] = match.group(2)
    return floors


def direct_url_dependencies(dependencies: tuple[str, ...]) -> list[str]:
    """Dependencies PyPI refuses (``name @ git+https://...``)."""
    return [dep for dep in dependencies if re.match(r"[A-Za-z0-9._-]+(\[[^\]]*\])?\s*@", dep)]


def plan_steps(state: State) -> list[str]:
    """Steps still to do, given what is already published."""
    if state.released:
        return ["wait_pypi"]
    steps = [] if state.tag_pushed else ["tests", "tag"]
    if not state.on_testpypi:
        steps.append("wait_testpypi")
    return [*steps, "verify", "confirm", "release", "wait_pypi"]


# ── External commands and services ────────────────────────────────────────────


def sh(*cmd: str, check: bool = True, capture: bool = True, env: dict[str, str] | None = None,
       cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd, capture_output=capture, text=True, encoding="utf-8", errors="replace", env=env, cwd=cwd, check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        raise ReleaseError(f"Command failed: {' '.join(cmd)}" + (f"\n{detail}" if detail else ""))
    return result


def pypi_versions(host: str, name: str) -> set[str]:
    request = urllib.request.Request(
        f"https://{host}/pypi/{name}/json", headers={"User-Agent": "re-ak-release-script"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return set(json.load(response)["releases"])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise ReleaseError(f"Cannot read {host}: {exc}") from exc
    except OSError as exc:
        raise ReleaseError(f"Cannot read {host}: {exc}") from exc


def remote_tag_commit(tag: str) -> str | None:
    """Commit of the tag on origin (the peeled line of an annotated tag comes last)."""
    out = sh("git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}").stdout.strip()
    return out.splitlines()[-1].split()[0] if out else None


def read_state(version: str, name: str) -> State:
    released = sh("gh", "release", "view", version, "--json", "tagName", check=False).returncode == 0
    return State(
        tag_pushed=remote_tag_commit(version) is not None,
        on_testpypi=version in pypi_versions(TEST_HOST, name),
        released=released,
    )


def latest_run(version: str, event: str) -> dict | None:
    out = sh("gh", "run", "list", "--workflow", WORKFLOW, "--branch", version, "--event", event, "--limit", "1",
             "--json", "databaseId,status,conclusion,url").stdout
    runs = json.loads(out or "[]")
    return runs[0] if runs else None


def wait_for_run(version: str, event: str, target: str) -> None:
    """Wait for the publish run; a run paused on a required reviewer is reported with its link, never as finished."""
    started = time.monotonic()
    last_status = None
    while True:
        run = latest_run(version, event)
        if run is None:
            if time.monotonic() - started > RUN_START_TIMEOUT_S:
                raise ReleaseError(f"No {WORKFLOW} run appeared for {version} ({event}). Check the Actions tab.")
        elif run["status"] == "completed":
            if run["conclusion"] == "success":
                print(f"  {target} run finished: {run['url']}")
                return
            print(f"  {target} run ended with '{run['conclusion']}': {run['url']}")
            print(f"  See why: gh run view {run['databaseId']} --log-failed")
            if input("  Rerun the failed jobs now? [y/N] ").strip().lower() != "y":
                raise ReleaseError("Fix the cause, then run this script again.")
            sh("gh", "run", "rerun", str(run["databaseId"]), "--failed")
            last_status = None
            time.sleep(POLL_SECONDS)
            continue
        elif run["status"] != last_status:
            last_status = run["status"]
            if last_status == "waiting":
                print(f"  ACTION NEEDED: approve the deployment (Review deployments): {run['url']}")
            else:
                print(f"  {target} run is {last_status}: {run['url']}")
        time.sleep(POLL_SECONDS)


def wait_on_index(host: str, name: str, version: str) -> None:
    deadline = time.monotonic() + INDEX_TIMEOUT_S
    while version not in pypi_versions(host, name):
        if time.monotonic() > deadline:
            raise ReleaseError(f"{name} {version} is not listed on {host} although the run succeeded.")
        time.sleep(POLL_SECONDS)
    print(f"  {name} {version} is listed on {host}.")


def verify_testpypi_package(project: Project) -> None:
    """Install from TestPyPI (dependencies from PyPI) in a throwaway venv and check the imported version."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory(prefix="release-check-", ignore_cleanup_errors=True) as tmp:
        venv = Path(tmp) / "venv"
        sh(sys.executable, "-m", "venv", str(venv), env=env, cwd=Path(tmp))
        python = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        spec = f"{project.name}=={project.version}"
        for attempt in range(1, INSTALL_ATTEMPTS + 1):
            print(f"  Installing {spec} from TestPyPI (attempt {attempt}/{INSTALL_ATTEMPTS})")
            result = sh(python, "-m", "pip", "install", "--no-cache-dir", "--quiet",
                        "-i", f"https://{TEST_HOST}/simple/", "--extra-index-url", f"https://{PROD_HOST}/simple/",
                        spec, check=False, env=env, cwd=Path(tmp))
            if result.returncode == 0:
                break
            if attempt == INSTALL_ATTEMPTS:
                raise ReleaseError(f"Could not install {spec} from TestPyPI:\n{result.stderr.strip()[-1500:]}")
            time.sleep(INSTALL_RETRY_S)
        code = (f"import {project.smoke_module}, {project.import_name} as m; "
                f"v = m.__version__; assert v == {project.version!r}, v; print(v)")
        result = sh(python, "-c", code, check=False, env=env, cwd=Path(tmp))
        if result.returncode != 0:
            raise ReleaseError(f"The installed package does not work:\n{result.stderr.strip()[-1500:]}")
    print(f"  Installed and imported {spec}.")


def run_tests(project: Project) -> None:
    """ruff and pytest on this checkout, not on whatever copy of the package the interpreter has installed."""
    env = {**os.environ, "PYTHONPATH": str(project.source_root), "QT_QPA_PLATFORM": "offscreen"}
    for name, cmd in (("ruff", ("ruff", "check", ".")), ("pytest", ("pytest", "-q"))):
        print(f"  Running {name}")
        if sh(sys.executable, "-m", *cmd, check=False, capture=False, env=env, cwd=project.root).returncode != 0:
            raise ReleaseError(f"{name} failed. Fix it through a pull request, then run this script again.")


def publish_release(project: Project, notes: str) -> None:
    with tempfile.TemporaryDirectory(prefix="release-notes-", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "notes.md"
        path.write_text(notes + "\n", encoding="utf-8")
        sh("gh", "release", "create", project.version, "--verify-tag",
           "--title", f"{project.name} {project.version}", "--notes-file", str(path))


# ── Checks ────────────────────────────────────────────────────────────────────


def preflight(project: Project, state: State) -> list[str]:
    """Everything that would stop the release, all at once."""
    version, failures = project.version, []

    if version_key(version) is None:
        failures.append(f"Version {version} is not x.y.z: the publish workflow only accepts such tags.")
    init_file = project.source_root / project.import_name / "__init__.py"
    in_init = init_version(init_file.read_text(encoding="utf-8")) if init_file.is_file() else None
    if in_init != version:
        failures.append(f"{init_file.name} says version {in_init}, pyproject.toml says {version}.")
    changelog = project.root / "CHANGELOG.md"
    if not changelog.is_file() or changelog_section(changelog.read_text(encoding="utf-8"), version) is None:
        failures.append(f"CHANGELOG.md has no content under '## {version}'.")
    if not (project.root / ".github" / "workflows" / WORKFLOW).is_file():
        failures.append(f".github/workflows/{WORKFLOW} is missing.")
    if not is_newer(version, pypi_versions(PROD_HOST, project.name)):
        failures.append(f"{version} is not newer than what is on PyPI: bump the version through a pull request.")
    for dep in direct_url_dependencies(project.dependencies):
        failures.append(f"PyPI refuses the direct URL dependency: {dep}")
    for name, floor in internal_floors(project.dependencies).items():
        if floor not in pypi_versions(PROD_HOST, name):
            failures.append(f"This release requires {name} {floor}, which is not on PyPI yet: release it first.")

    if sh("git", "status", "--porcelain", "--untracked-files=no").stdout.strip():
        failures.append("There are uncommitted changes to tracked files.")
    sh("git", "fetch", "--quiet", "origin", DEFAULT_BRANCH)
    head = sh("git", "rev-parse", "HEAD").stdout.strip()
    tag_commit = remote_tag_commit(version)
    if tag_commit is not None:
        # Resuming: the tag decides what gets released, main may have moved on since.
        if tag_commit != head:
            failures.append(f"Tag {version} is on {tag_commit[:7]}, this checkout on {head[:7]}. "
                            f"To resume: git switch --detach {version}")
    else:
        branch = sh("git", "branch", "--show-current").stdout.strip()
        if branch != DEFAULT_BRANCH:
            failures.append(f"Run this from {DEFAULT_BRANCH} (currently on '{branch or 'detached HEAD'}').")
        elif head != sh("git", "rev-parse", f"origin/{DEFAULT_BRANCH}").stdout.strip():
            failures.append(f"Local {DEFAULT_BRANCH} differs from origin: git pull --ff-only")
        if state.on_testpypi:
            failures.append(f"{version} is on TestPyPI but has no tag, and it cannot be uploaded twice: "
                            "bump the version through a pull request.")
    return failures


def confirm(project: Project, notes: str) -> None:
    print(f"\n--- {project.name} {project.version} release notes ---\n{notes}\n---")
    answer = input(f"Publish {project.name} {project.version} to PyPI? This cannot be undone. Type 'yes': ")
    if answer.strip().lower() != "yes":
        raise ReleaseError("Stopped before PyPI. Run this script again to continue from here.")


# ── Entry point ───────────────────────────────────────────────────────────────


def release(root: Path, *, dry_run: bool = False, skip_tests: bool = False) -> int:
    for tool in ("git", "gh"):
        if shutil.which(tool) is None:
            raise ReleaseError(f"{tool} is not installed or not on PATH.")
    sh("gh", "auth", "status")

    project = read_project(root)
    print(f"{project.name} {project.version}")
    if project.version in pypi_versions(PROD_HOST, project.name):
        print("Already on PyPI, nothing to do. To release again, bump the version through a pull request.")
        return 0

    state = read_state(project.version, project.name)
    failures = preflight(project, state)
    if failures:
        print("Cannot release:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    untracked = sh("git", "status", "--porcelain").stdout.strip()
    if untracked:
        print("Note: untracked files are present; they are not part of the release.")

    steps = plan_steps(state)
    if dry_run:
        print("Checks passed. Remaining steps: " + ", ".join(steps))
        return 0

    notes = changelog_section((root / "CHANGELOG.md").read_text(encoding="utf-8"), project.version) or ""
    for step in steps:
        if step == "tests":
            if skip_tests:
                print("Skipping ruff and pytest (--skip-tests).")
            else:
                print("Testing this checkout")
                run_tests(project)
        elif step == "tag":
            print(f"Tagging {project.version} (uploads to TestPyPI)")
            sh("git", "tag", "-f", project.version)
            sh("git", "push", "origin", f"refs/tags/{project.version}")
        elif step == "wait_testpypi":
            print("Waiting for the TestPyPI upload")
            wait_for_run(project.version, "push", "TestPyPI")
            wait_on_index(TEST_HOST, project.name, project.version)
        elif step == "verify":
            print("Testing the TestPyPI package")
            verify_testpypi_package(project)
        elif step == "confirm":
            confirm(project, notes)
        elif step == "release":
            print(f"Creating the GitHub release {project.version} (uploads to PyPI)")
            publish_release(project, notes)
        elif step == "wait_pypi":
            print("Waiting for the PyPI upload")
            wait_for_run(project.version, "release", "PyPI")
            wait_on_index(PROD_HOST, project.name, project.version)

    print(f"\nReleased: pip install -U {project.name}=={project.version}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Release this repository's package on TestPyPI, then PyPI.")
    parser.add_argument("--dry-run", action="store_true", help="run the checks and print the remaining steps")
    parser.add_argument("--skip-tests", action="store_true", help="skip ruff and pytest")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")  # type: ignore[attr-defined]  # changelogs hold non-cp1252 characters
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    try:
        return release(root, dry_run=args.dry_run, skip_tests=args.skip_tests)
    except ReleaseError as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Run this script again to continue from where it stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
