"""
EDA / GSR arousal processing for Shimmer data.

Computes arousal from EDA (GSR). EDA is read from ``rawData/shimmer.csv``
(same column layout as legacy ``rawShimmer_0.csv``): timestamp in column 0,
EDA in column 4.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram

from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError

# Module-level constants
SHAPE_PARAM = 0.2
PLACEMENT_PARAM = -18
KERNEL_LENGTH_TIME = 45
OFFSET = -2
DT = 0.5
SAMPLING_RATE = 1 / DT
FS = 51.2  # Sampling frequency

# Shimmer CSV: column 0 = timestamp (ms), column 4 = EDA / GSR (same as rawShimmer_0.csv)
_SHIMMER_TIMESTAMP_COL = 0
_SHIMMER_EDA_COL = 4


def _load_shimmer_eda_signal(rec_path):
    """
    Load timestamp + EDA from ``rawData/shimmer.csv`` or legacy ``rawShimmer_0.csv``.

    Returns a two-column DataFrame (columns 0 = time in seconds from start, 1 = EDA),
    or None if no usable file is present.
    """
    raw_dir = os.path.join(rec_path, "rawData")
    for name in ("shimmer.csv", "rawShimmer_0.csv"):
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, header=None)
        if df.shape[1] <= _SHIMMER_EDA_COL:
            printWarning(
                f"[edaProcessor] {path} has fewer than {_SHIMMER_EDA_COL + 1} columns; "
                "cannot read EDA (column 4)."
            )
            continue
        signal = df[[_SHIMMER_TIMESTAMP_COL, _SHIMMER_EDA_COL]].copy()
        signal.columns = [0, 1]
        t0 = signal[0].iloc[0]
        signal[0] = (signal[0] - t0) / 1000.0
        return signal

    printWarning(
        f"[edaProcessor] No Shimmer EDA file in {raw_dir} "
        "(expected shimmer.csv or rawShimmer_0.csv). Shimmer was likely not used for this recording."
    )
    return None


def validateSignalQuality(eda):
    mean_signal = np.mean(eda)
    var_signal = np.var(eda)

    if mean_signal < 0.0009 or var_signal > 1:
        printInfo(f"mean_signal < 0.0009: {mean_signal}")
        printInfo(f"var_signal > 1: {var_signal}")
        return False
    return True


def computeArousal(recPath, show=False):
    """
    Compute arousal from GSR data using skin conductance responses (SCRs).

    Args:
        recPath: Path to the recording directory
        show: Whether to display plots (default: False)
    """
    printInfo("[physioTools] Computing Arousal")

    # Skip if already done (caching)
    arousal_path = os.path.join(recPath, "results", "Arousal.csv")
    if os.path.isfile(arousal_path):
        printInfo("[physioTools] Arousal already computed, using cached results")
        return

    signal = _load_shimmer_eda_signal(recPath)
    if signal is None:
        return

    eda_features_dir = os.path.join(recPath, "features", "eda")
    os.makedirs(eda_features_dir, exist_ok=True)

    filtered_signal = conditionSignal(signal[1])

    eda = 1 / filtered_signal

    if not validateSignalQuality(eda):
        printWarning(f"[edaProcessor] {recPath} EDA failed QA")

        plt.figure(figsize=(10, 5))

        plt.plot(eda, label='scrs', color='blue')
        plt.legend()

        plt.savefig(os.path.join(eda_features_dir, "eda_rejected.png"))
        plt.close()

        return

    scrs = extract_scrs(eda, np.mean(eda)/10)

    plt.figure(figsize=(10, 5))

    for index in scrs:
        plt.axvline(x=index, color='k')

    plt.plot(eda, label='scrs', color='blue')
    plt.legend()

    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(eda_features_dir, "eda_wSCRs.png"))
        plt.close()

    duration = signal[0].values[-1]
    eventsTime = signal[0].values[scrs]
    timeline = np.arange(0, duration, 0.5)
    arousal = np.zeros(timeline.shape)

    addKernel(arousal, eventsTime)

    arousal_df = pd.DataFrame()
    arousal_df["Timestamp"] = timeline*1000
    arousal_df["Arousal"] = arousal
    arousal_df.to_csv(arousal_path, index=False)

    # Combine them as columns
    arousalRaw_df = pd.DataFrame(data=np.column_stack((signal[0].values, eda)), columns=["Timestamp", "Arousal"])

    # Simple resampling without external dependency
    # TODO: Implement resample function or use pandas resample
    # arousalRaw_df = resample(arousalRaw_df,"500ms")

    arousalRaw_df.to_csv(os.path.join(recPath, "results", "ArousalRaw.csv"), index=False)
    printInfo("[physioTools] Arousal computation completed")


def extract_scrs(signal, threshold):
    minima = []
    maxima = []
    n = len(signal)

    # Step 1: Identify local minima and maxima
    for i in range(1, n - 1):
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            minima.append(i)
        elif signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            maxima.append(i)

    # Step 2: Remove first maximum if it's the first extremum
    if minima and maxima:
        if maxima[0] < minima[0]:
            maxima.pop(0)

    # Step 3: Remove last minimum if it's the last extremum
    if minima and maxima:
        if minima[-1] > maxima[-1]:
            minima.pop()

    # Step 4: Pair minimums with maximums and filter by threshold
    accepted_minima = []
    for i in range(min(len(minima), len(maxima))):
        min_idx = minima[i]
        max_idx = maxima[i]
        diff = signal[max_idx] - signal[min_idx]

        if diff >= threshold:
            accepted_minima.append(min_idx)

    # Step 5: Return the indices of accepted minima
    return accepted_minima

def conditionSignal(signal, fs=FS):
    # Determine the cutoff frequency
    frequencies, power_spectrum = periodogram(signal, fs)
    cutoff = 0.1 / 2  # Using a lower frequency for cutoff

    # Apply the low-pass filter
    filtered_signal = lowpass_filter(signal, cutoff, fs)

    return filtered_signal


def sigmoid(x, a, b):
    s = 1 - 1 / (1 + np.exp((-a * (x + b))))
    return s


tt = np.arange(OFFSET, KERNEL_LENGTH_TIME, DT)
EDA_SIG_KERNEL = sigmoid(tt, SHAPE_PARAM, PLACEMENT_PARAM)

def addKernel(EDAEventsTime, events):

    for i in range(events.shape[0]):
        eventPos = int(events[i]) * 2

        start = int(eventPos + OFFSET * SAMPLING_RATE)
        end = start + EDA_SIG_KERNEL.shape[0]

        # Clamp to the valid signal range
        sig_start = max(0, start)
        sig_end = min(EDAEventsTime.shape[0], end)

        if sig_start >= sig_end:
            continue

        # Mirror the clamping on the kernel side
        ker_start = sig_start - start
        ker_end = ker_start + (sig_end - sig_start)

        EDAEventsTime[sig_start:sig_end] += EDA_SIG_KERNEL[ker_start:ker_end]



# Assuming your signal is stored in the variable 'signal' and sampled at 51.2 Hz

# Define a Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def loadArousal(basepath):
    """
    Load arousal data from CSV file.

    Args:
        basepath: Base path to the recording directory

    Returns:
        numpy array containing arousal data
    """
    arousal_path = os.path.join(basepath, "results", "Arousal.csv")

    if not os.path.isfile(arousal_path):
        printWarning(f"[physioTools] Arousal file not found: {arousal_path}")
        return None

    try:
        df = pd.read_csv(arousal_path)
        return df.values
    except Exception as e:
        printError(f"[physioTools] Error loading arousal data: {e}")
        return None
