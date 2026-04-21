import numpy as np


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
