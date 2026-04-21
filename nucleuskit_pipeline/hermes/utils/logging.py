"""
Hermes Pipeline Logging Utilities

Provides logging functions for the Hermes analytics pipeline,
including file-based logging for processing steps.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os
from nucleuskit_pipeline.logging_utils import printWarning


def printToFile(recpath, line):
    """
    Write a line to the processing log file.
    
    Creates a timestamped log entry in the session's results directory.
    
    Args:
        recpath: Base path to the session recording directory
        line: Line of text to append to the log
    """
    try:
        log_path = os.path.join(recpath, 'results', 'processingLog.txt')
        with open(log_path, 'a') as processingLog:
            processingLog.write(line + "\n")
    except Exception as e:
        printWarning(f"[logging] Cannot write to processingLog.txt: {e}")
