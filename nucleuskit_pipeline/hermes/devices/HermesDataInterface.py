"""
Hermes Data Interface

Encapsulates the data interface of a recording made with the Hermes device.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os
import traceback

import numpy as np
import pandas as pd

from nucleuskit_pipeline.hermes.HermesConstants import HermesConstants
from nucleuskit_pipeline.hermes.processor.utils.fileInterface import loadtxt_drop_last_if_incomplete
from nucleuskit_pipeline.logging_utils import printError, printInfo


def _load_exg_numeric_csv(filepath):
    """
    Load raw EXG/EEG matrix from CSV (headerless numeric rows, or skip one header row).

    Returns:
        numpy array or None on failure
    """
    try:
        return np.loadtxt(filepath, delimiter=",")
    except ValueError:
        try:
            return np.loadtxt(filepath, delimiter=",", skiprows=1)
        except Exception as e:
            printError(f"[HermesDataInterface] Could not load EXG file (with header skip) {filepath}: {e}")
            printError(f"[HermesDataInterface] Traceback:\n{traceback.format_exc()}")
            return None
    except Exception as e:
        printError(f"[HermesDataInterface] Could not load EXG file {filepath}: {e}")
        printError(f"[HermesDataInterface] Traceback:\n{traceback.format_exc()}")
        return None


def loadEXG(rec_path, re_reference=False):
    """
    Load and optionally re-reference EXG (EMG/EEG) signals from a Hermes recording raw folder.

    Args:
        rec_path: Path to the recording directory
        re_reference: If True, subtract the average of EAR_L and EAR_R from all channels

    Returns:
        Tuple (timestamps, exg_data) or None on failure
    """
    valid_filenames = ["rawEEG_0.csv", "eeg.tmp", "eeg.csv", "eegRec_0.csv"]
    filename = None

    for tmp in valid_filenames:
        if os.path.exists(os.path.join(rec_path, "rawData", tmp)):
            filename = tmp
            break

    if filename is None:
        printError(f"[HermesDataInterface] No valid EXG file found in {os.path.join(rec_path, 'rawData')}")
        printError(f"[HermesDataInterface] Looked for: {valid_filenames}")
        return None

    filepath = os.path.join(rec_path, "rawData", filename)
    exg_data = _load_exg_numeric_csv(filepath)

    if exg_data is None:
        printError(f"[HermesDataInterface] Failed to load EXG data from {filepath}")
        return None

    if exg_data.ndim < 2 or exg_data.shape[1] < 2:
        printError(
            f"[HermesDataInterface] EXG data has unexpected shape {exg_data.shape} (expected at least 2 columns)"
        )
        return None

    try:
        timestamps = exg_data[:, 0]
        timestamps -= timestamps[0]
        exg_data = exg_data[:, 1:]
    except Exception as e:
        printError(f"[HermesDataInterface] Failed to extract timestamps from EXG data: {e}")
        printError(f"[HermesDataInterface] Traceback:\n{traceback.format_exc()}")
        return None

    if re_reference:
        try:
            ref_signal = exg_data[:, HermesConstants.CHANNELS["EAR_R"]]/ 2

            for i in range(exg_data.shape[1]):
                exg_data[:, i] -= ref_signal
        except Exception as e:
            printError(f"[HermesDataInterface] Re-referencing failed: {e}")
            printError(f"[HermesDataInterface] Traceback:\n{traceback.format_exc()}")
            return None

    return timestamps, exg_data


class HermesDataInterface:

    SAMPLING_RATE = 250  # Hz

    def __init__(self, recPath):
        self.recPath = recPath

    def getEEG(self):
        """
        Load and return the EEG data for this recording as a DataFrame.

        Searches for a raw EEG file, strips corrupted timestamps, removes
        saturated electrode values, and assembles the channel layout:
            AF7       : channel 2 - channel 5 / 2
            AF8       : channel 1 - channel 5 / 2
            Temporal  : channel 5 (raw)
            LeftHemi  : channel 2 (raw)
            RightHemi : channel 1 - channel 5 (differential)

        Returns:
            pandas DataFrame with one column per channel, or None if no data
            file is found.
        """
        potentialFilenames = ["rawEEG_0.csv", "eeg.tmp", "eeg.csv", "eegRec_0.csv"]
        data = None
        tried = []

        for filename in potentialFilenames:
            filepath = os.path.join(self.recPath, "rawData", filename)
            tried.append(filepath)
            if os.path.isfile(filepath):
                printInfo(f"[HermesDataInterface] Found EEG file: {filepath}")
                try:
                    data = loadtxt_drop_last_if_incomplete(filepath)
                    break
                except Exception as e:
                    printError(f"[HermesDataInterface] Failed to load {filename}: {e}")
                    printError(f"[HermesDataInterface] Traceback:\n{traceback.format_exc()}")

        if data is None:
            printError("[HermesDataInterface] No EEG data could be loaded.")
            printError(f"[HermesDataInterface] Looked for: {tried}")
            return None

        printInfo(f"[HermesDataInterface] EEG raw data shape: {data.shape}")

        if data.shape[1] < 6:
            printError(f"[HermesDataInterface] EEG data has only {data.shape[1]} columns — "
                       f"expected at least 6. File may be corrupt or use an unexpected format.")
            return None

        # Remove saturated values (disconnected electrodes)
        data[abs(abs(data) - 187500) < 0.1] = np.nan

        return pd.DataFrame({
            'AF7':      data[:, 2] - data[:, 5] / 2,
            'AF8':      data[:, 1] - data[:, 5] / 2,
            'Temporal': data[:, 5],
            'LeftHemi': data[:, 2],
            'RightHemi': data[:, 1] - data[:, 5],
        })
