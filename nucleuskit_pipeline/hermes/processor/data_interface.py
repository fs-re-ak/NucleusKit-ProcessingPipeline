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

from nucleuskit_pipeline.hermes.constants import HermesConstants
from nucleuskit_pipeline.shared.file_interface import (
    loadtxt_drop_last_if_incomplete,
    normalise_timestamps_to_seconds,
)
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
        timestamps = normalise_timestamps_to_seconds(exg_data[:, 0])
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

    SAMPLING_RATE = HermesConstants.SAMPLING_RATE

    def __init__(self, recPath):
        self.recPath = recPath

    def getEEG(self, raw_preprocess_fn=None):
        """
        Load and return the EEG data for this recording together with the
        original hardware timestamps.

        Searches for a raw EEG file, strips corrupted timestamps, removes
        saturated electrode values, and assembles the channel layout using a
        midpoint re-reference against the T9/T10 differential (EAR_R, col 5):

            AF7       : col2 - col5 / 2  (left frontal, half-referenced)
            AF8       : col1 - col5 / 2  (right frontal, half-referenced)
            T9        : -col5 / 2        (left temporal, reference side mirrored)
            T10       :  col5 / 2        (right temporal, sensing side)
            LeftHemi  : col2             (raw left hemisphere)
            RightHemi : col1 - col5      (right hemisphere differential)

        T9 is the hardware reference electrode (left ear) and T10 is the
        sensing electrode (right ear / EAR_R).  The differential col5 encodes
        T10 - T9, so after the midpoint split T9 = -col5/2 and T10 = col5/2.

        Returns:
            Tuple (timestamps, df) where ``timestamps`` is a 1-D numpy array
            of hardware-clock timestamps in seconds from recording start and
            ``df`` is a pandas DataFrame with one column per channel.
            Returns ``(None, None)`` if no data file is found.
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
            return None, None

        printInfo(f"[HermesDataInterface] EEG raw data shape: {data.shape}")

        if data.shape[1] < 6:
            printError(f"[HermesDataInterface] EEG data has only {data.shape[1]} columns — "
                       f"expected at least 6. File may be corrupt or use an unexpected format.")
            return None, None

        timestamps = normalise_timestamps_to_seconds(data[:, 0])

        # Remove saturated values (disconnected electrodes)
        data[abs(abs(data) - 187500) < 0.1] = np.nan

        if raw_preprocess_fn is not None:
            from nucleuskit_pipeline.hermes.realtime.nlms_filter import nlms_nan_safe
            data[:, 1:] = nlms_nan_safe(raw_preprocess_fn, data[:, 1:])

        mid_ref = data[:, 5] / 2.0
        df = pd.DataFrame({
            'AF7':      data[:, 2] - mid_ref,
            'AF8':      data[:, 1] - mid_ref,
            'T9':      -mid_ref,               # left temporal (reference mirrored)
            'T10':      mid_ref,               # right temporal (sensing side)
            'LeftHemi': data[:, 2],
            'RightHemi': data[:, 1] - data[:, 5],
        })
        return timestamps, df

    def getReferenceChannels(self):
        """
        Load the four EMG reference channels that are recorded by the hardware
        but not used for EEG derivation.

        These channels share the same noise sources as the EEG channels (temporalis
        muscle via CHEEK_R/CHEEK_L, frontal muscle via BROW_L, and broadband
        common-mode noise via NOSE) and serve as regressors for noise removal.

        Raw column mapping (0-indexed after stripping the timestamp column):
            CHEEK_R  → col 2  (right cheek EMG)
            CHEEK_L  → col 3  (left cheek EMG)
            BROW_L   → col 6  (left brow EMG)
            NOSE     → col 7  (nose common reference)

        AFz (col 5) is deliberately excluded — it is an EEG channel and
        regressing it out would remove real brain signal.

        Returns:
            Tuple (timestamps, ref_df) where ``timestamps`` is a 1-D numpy array
            (seconds from recording start) and ``ref_df`` is a DataFrame with
            columns ['CHEEK_R', 'CHEEK_L', 'BROW_L', 'NOSE'].
            Returns ``(None, None)`` if no data file is found or loading fails.
        """
        potentialFilenames = ["rawEEG_0.csv", "eeg.tmp", "eeg.csv", "eegRec_0.csv"]
        data = None

        for filename in potentialFilenames:
            filepath = os.path.join(self.recPath, "rawData", filename)
            if os.path.isfile(filepath):
                printInfo(f"[HermesDataInterface] getReferenceChannels: loading {filepath}")
                try:
                    data = loadtxt_drop_last_if_incomplete(filepath)
                    break
                except Exception as e:
                    printError(f"[HermesDataInterface] getReferenceChannels: failed to load {filename}: {e}")

        if data is None:
            printError("[HermesDataInterface] getReferenceChannels: no EEG data file found")
            return None, None

        if data.shape[1] < 9:
            printError(
                f"[HermesDataInterface] getReferenceChannels: data has only {data.shape[1]} columns "
                f"— expected at least 9 (timestamp + 8 EXG channels). Cannot extract reference channels."
            )
            return None, None

        timestamps = normalise_timestamps_to_seconds(data[:, 0])

        # Invalidate saturated (disconnected) samples
        data[abs(abs(data) - 187500) < 0.1] = np.nan

        # Raw columns 3, 4, 7, 8 = CHEEK_R, CHEEK_L, BROW_L, NOSE
        ref_df = pd.DataFrame({
            'CHEEK_R': data[:, 3],
            'CHEEK_L': data[:, 4],
            'BROW_L':  data[:, 7],
            'NOSE':    data[:, 8],
        })

        return timestamps, ref_df
