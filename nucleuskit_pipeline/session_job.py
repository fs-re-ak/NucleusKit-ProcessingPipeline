"""
Minimal session job DTO for local Nucleus-Kit pipeline processing (no REST).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionJob:
    sessionID: Optional[str] = None
    path: Optional[str] = None
    jobType: str = "PROCESS"
    pipeline: str = "hermes"
    screenID: Optional[str] = None
    pov_config_json: Optional[str] = None  # optional path merged after cwd default JSON


def session_job_from_folder(folder: str, pov_config_json: Optional[str] = None) -> SessionJob:
    folder = os.path.abspath(os.path.expanduser(folder))
    norm = os.path.normpath(folder)
    session_id = os.path.basename(norm)
    sep = os.path.sep
    path = norm if norm.endswith(sep) else norm + sep
    return SessionJob(sessionID=session_id, path=path, pov_config_json=pov_config_json)
