"""
GPS Processor

Processes GPS tracking data from sessions.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os

import numpy as np
import pandas as pd

from nucleuskit_pipeline.logging_utils import printInfo, printWarning

GPS_HDR = "[GPSProcessor] "

_GPS_FILENAMES = ["gps.csv", "gps.tmp"]
_GRID_STEP_S   = 0.5   # shared 2 Hz / 0.5 s pipeline timebase


def processGPS(recpath):
    """
    Process GPS tracking data.

    Loads the raw GPS CSV (no header: unix_timestamp_s, latitude, longitude),
    normalises timestamps to session-relative seconds, resamples to the shared
    2 Hz / 0.5 s grid via linear interpolation, and writes results/gpsDf.csv.

    Args:
        recpath: Path to the recording directory
    """
    printInfo(f"{GPS_HDR}Processing GPS data")

    out_path = os.path.join(recpath, "results", "gpsDf.csv")
    if os.path.exists(out_path):
        printInfo(f"{GPS_HDR}GPS already processed, using cached results")
        return

    gps_file = None
    for filename in _GPS_FILENAMES:
        candidate = os.path.join(recpath, "rawData", filename)
        if os.path.isfile(candidate):
            gps_file = candidate
            break

    if gps_file is None:
        printWarning(f"{GPS_HDR}No GPS file found, tested: {_GPS_FILENAMES}")
        return

    printInfo(f"{GPS_HDR}Loading {os.path.basename(gps_file)}")
    try:
        raw = pd.read_csv(
            gps_file,
            header=None,
            names=["unix_ts", "Latitude", "Longitude"],
        )
    except Exception as exc:
        printWarning(f"{GPS_HDR}Failed to read GPS file: {exc}")
        return

    if len(raw) < 2:
        printWarning(f"{GPS_HDR}GPS file contains fewer than 2 rows — skipping")
        return

    t_raw = raw["unix_ts"].to_numpy(dtype=float)
    t_rel = t_raw - t_raw[0]
    lat   = raw["Latitude"].to_numpy(dtype=float)
    lon   = raw["Longitude"].to_numpy(dtype=float)

    grid = np.arange(0.0, t_rel[-1] + _GRID_STEP_S, _GRID_STEP_S)

    lat_grid = np.interp(grid, t_rel, lat, left=np.nan, right=np.nan)
    lon_grid = np.interp(grid, t_rel, lon, left=np.nan, right=np.nan)

    result = pd.DataFrame({
        "Timestamp": grid,
        "Latitude":  lat_grid,
        "Longitude": lon_grid,
    })

    result.to_csv(out_path, index=False, float_format="%.7f", na_rep="NULL")
    printInfo(
        f"{GPS_HDR}GPS processing completed — "
        f"{len(result)} rows @ 2 Hz written to {out_path}"
    )
