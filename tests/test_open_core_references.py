"""Non-regression against the open-core reference outputs (re-ak-core, Phase 2).

Runs the current NucleusKit processing steps on deterministic synthetic sessions generated
by ``re-ak-core/tests/synthetic`` and compares the outputs with the references recorded in
``re-ak-core/tests/references/nk``. A failure means NucleusKit's results changed: either fix
the regression or, if the change is intended, re-record the references in re-ak-core and
report it in the PR (the core must follow).

Needs a ``re-ak-core`` checkout next to this repository (or ``RE_AK_CORE=<path>``); skipped
otherwise. Runs two cases by default (about 30 s); set ``RE_AK_REFERENCE_CASES=all`` or a
comma-separated list for more.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORE = os.environ.get("RE_AK_CORE", os.path.join(os.path.dirname(_REPO), "re-ak-core"))
_DEFAULT_CASES = ["baseline", "emg_rms_threshold"]

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(_CORE, "tools", "refkit.py")),
    reason="re-ak-core checkout not found (set RE_AK_CORE)",
)


def _cases():
    sys.path.insert(0, os.path.join(_CORE, "tools"))
    import refkit

    wanted = os.environ.get("RE_AK_REFERENCE_CASES", "")
    if wanted == "all":
        return refkit, [c.name for c in refkit.CASES]
    return refkit, wanted.split(",") if wanted else _DEFAULT_CASES


def test_outputs_match_open_core_references(tmp_path) -> None:
    refkit, cases = _cases()
    failures = refkit.check_system("nk", _REPO, str(tmp_path), cases)
    report = "\n".join(
        f"{key}: {d.output}: {d.detail}" for key, diffs in failures.items() for d in diffs
    )
    assert not failures, f"NucleusKit outputs differ from the open-core references:\n{report}"
