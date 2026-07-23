import numpy as np

from nucleuskit_toolkit.logging_utils import printInfo

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


def has_text_header(filepath, delimiter=","):
    """Return True if the first non-empty CSV line cannot be parsed as all-numeric values.

    Used by raw-data loaders to auto-detect an optional header row and skip it,
    while keeping the same positional column indices for legacy headerless files.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str, optional
        Column delimiter (default is ',').

    Returns
    -------
    bool
        ``True`` when the first non-empty line contains at least one
        non-numeric field (i.e. a text header); ``False`` when every field
        can be parsed as a float (i.e. the file starts with data).
    """
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    [float(x) for x in line.split(delimiter)]
                    return False
                except ValueError:
                    return True
    return False


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
