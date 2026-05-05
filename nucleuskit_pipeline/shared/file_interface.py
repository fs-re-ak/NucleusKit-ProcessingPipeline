import numpy as np

from nucleuskit_pipeline.logging_utils import printInfo

# Epoch-millisecond timestamps for any date from 2001 onward are above 1e10.
# Epoch-second timestamps for the same range are below 2e9.
# A threshold of 1e10 unambiguously separates the two for all foreseeable dates.
_MS_EPOCH_THRESHOLD = 1e10


def normalise_timestamps_to_seconds(timestamps):
    """Return hardware epoch timestamps zeroed to recording start, in seconds.

    Both Shimmer and Hermes hardware record wall-clock epoch timestamps.
    Legacy recordings store them in **milliseconds**; recent recordings store
    them in **seconds**. The timebase is detected automatically from the raw
    magnitude:

    - epoch-ms values in 2026 are ~1.746 × 10¹²  (> 1e10)
    - epoch-s  values in 2026 are ~1.746 × 10⁹   (< 1e10)

    Parameters
    ----------
    timestamps : array-like
        Raw timestamp column as read from the hardware CSV (monotonically
        increasing epoch values, either ms or s).

    Returns
    -------
    np.ndarray
        Timestamps in **seconds**, zeroed so that ``t[0] == 0``.
    """
    ts = np.asarray(timestamps, dtype=float)
    if ts[0] > _MS_EPOCH_THRESHOLD:
        printInfo("[timestamps] Epoch-millisecond timestamps detected — converting to seconds.")
        ts = ts / 1000.0
    return ts - ts[0]


def loadtxt_drop_last_if_incomplete(filepath, delimiter=","):
    """
    Load a CSV file into a NumPy array.

    Handles three common issues:
    - Incomplete last line  : retries after dropping the last row.
    - Text header row       : retries after dropping the first row.
    - Both simultaneously   : retries after dropping both first and last rows.

    Raises ValueError for any other loading issue.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str, optional
        Column delimiter (default is ',').

    Returns
    -------
    np.ndarray
        The loaded data array.
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Attempt 1: load as-is
    try:
        return np.loadtxt(lines, delimiter=delimiter)
    except ValueError:
        pass

    # Attempt 2: incomplete last line — drop it
    try:
        return np.loadtxt(lines[:-1], delimiter=delimiter)
    except ValueError:
        pass

    # Attempt 3: text header in first row — skip it
    try:
        return np.loadtxt(lines[1:], delimiter=delimiter)
    except ValueError:
        pass

    # Attempt 4: header AND incomplete last line — skip both
    try:
        return np.loadtxt(lines[1:-1], delimiter=delimiter)
    except ValueError:
        raise ValueError(
            f"Failed to load '{filepath}' — tried with/without header and with/without last line."
        )
