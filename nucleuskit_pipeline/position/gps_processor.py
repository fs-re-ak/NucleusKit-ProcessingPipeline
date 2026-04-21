"""
GPS Processor

Processes GPS tracking data from sessions.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os

from nucleuskit_pipeline.logging_utils import printInfo, printWarning


def processGPS(recpath):
    """
    Process GPS tracking data.

    Args:
        recpath: Path to the recording directory
    """
    printInfo("[GPSProcessor] Processing GPS data")

    # Skip if already done (caching)
    gps_path = os.path.join(recpath, "results", "gpsDf.csv")
    if os.path.exists(gps_path):
        printInfo("[GPSProcessor] GPS already processed, using cached results")
        return

    # TODO: Implement GPS processing
    printWarning("[GPSProcessor] GPS processing not yet implemented")
