"""
EMG Utilities

Functions for loading and processing EMG signals from Hermes recordings.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os
import sys
import traceback
import numpy as np
from scipy import signal
from scipy.signal import welch
from scipy.interpolate import interp1d
from nucleuskit_pipeline.logging_utils import printWarning, printError

# Optional: EntropyHub for sample entropy computation
try:
    import EntropyHub as EH
    ENTROPYHUB_AVAILABLE = True
except ImportError:
    ENTROPYHUB_AVAILABLE = False
    printWarning("[emgUtils] EntropyHub not available - sample entropy computation disabled")


def loadEEGRaw(filename):
    """
    Load raw EEG data from CSV file.

    Handles both headerless CSVs (all-numeric rows) and CSVs whose first row
    contains column labels.

    Args:
        filename: Path to the EEG CSV file

    Returns:
        numpy array with EEG data, or None on failure
    """
    try:
        data = np.loadtxt(filename, delimiter=",")
        return data
    except ValueError:
        # First row is likely a text header — skip it and retry
        try:
            data = np.loadtxt(filename, delimiter=",", skiprows=1)
            return data
        except Exception as e:
            printError(f"[emgUtils] Could not load EEG file (with header skip) {filename}: {e}")
            printError(f"[emgUtils] Traceback:\n{traceback.format_exc()}")
            return None
    except Exception as e:
        printError(f"[emgUtils] Could not load EEG file {filename}: {e}")
        printError(f"[emgUtils] Traceback:\n{traceback.format_exc()}")
        return None


def conditionEMG(emgRec, reReference=False, filterOrder=4, bandwidth=[15, 45], nyquistFreq=125):
    """
    Apply bandpass filtering to EMG signals.

    Args:
        emgRec: EMG signal array (samples x channels)
        reReference: Whether to subtract the channel mean
        filterOrder: Butterworth filter order
        bandwidth: [low, high] cutoff frequencies in Hz
        nyquistFreq: Nyquist frequency in Hz

    Returns:
        Filtered EMG array, or None on failure
    """
    try:
        if reReference:
            emgRec -= np.average(emgRec, axis=1)[:, None]

        for i in range(emgRec.shape[1]):
            b, a = signal.butter(filterOrder, bandwidth[1] / nyquistFreq, btype='lowpass')
            low_passed = signal.filtfilt(b, a, emgRec[:, i])
            emgRec[:, i] = signal.filtfilt(b, a, low_passed)

            b, a = signal.butter(filterOrder, bandwidth[0] / nyquistFreq, btype='highpass')
            high_passed = signal.filtfilt(b, a, emgRec[:, i])
            emgRec[:, i] = signal.filtfilt(b, a, high_passed)

        return emgRec

    except Exception as e:
        printError(f"[emgUtils] conditionEMG failed: {e}")
        printError(f"[emgUtils] Traceback:\n{traceback.format_exc()}")
        return None


def extractEnveloppe(emgRec, filterOrder=4, enveloppeFc=2.5, nyquistFreq=125):
    """
    Extract the amplitude envelope of EMG signals via rectification and low-pass filtering.

    Args:
        emgRec: EMG signal array (samples x channels)
        filterOrder: Butterworth filter order
        enveloppeFc: Envelope low-pass cutoff frequency in Hz
        nyquistFreq: Nyquist frequency in Hz

    Returns:
        Envelope array, or None on failure
    """
    try:
        emgRec = np.abs(emgRec)

        for i in range(emgRec.shape[1]):
            b, a = signal.butter(filterOrder, enveloppeFc / nyquistFreq, btype='lowpass')
            low_passed = signal.filtfilt(b, a, emgRec[:, i])
            emgRec[:, i] = signal.filtfilt(b, a, low_passed)

        return emgRec

    except Exception as e:
        printError(f"[emgUtils] extractEnveloppe failed: {e}")
        printError(f"[emgUtils] Traceback:\n{traceback.format_exc()}")
        return None


def extractPowerBands(data, sampling_rate=250, window_length=250, overlap=0.5):
    power_bands = []
    channels = data.shape[1]

    # Define the frequency bands
    frequency_bands = [(15, 20), (20, 25), (25, 30), (30, 35), (35, 45)]

    # Calculate the number of samples per window
    window_samples = int(window_length)

    # Calculate the overlap in samples
    overlap_samples = int(window_samples * overlap)

    # Compute the frequency axis
    freqs, _ = welch(data[:, 0], fs=sampling_rate, nperseg=window_samples, noverlap=overlap_samples)

    # Iterate over each channel
    for channel in range(channels):
        channel_power_bands = []

        # Iterate over the defined frequency bands
        for band in frequency_bands:
            _, psd = welch(data[:, channel], fs=sampling_rate, nperseg=window_samples, noverlap=overlap_samples)

            lower_freq, upper_freq = band
            band_indices = np.where((freqs >= lower_freq) & (freqs <= upper_freq))
            band_power = np.trapz(psd[band_indices], dx=freqs[1])

            channel_power_bands.append(band_power)

        power_bands.append(channel_power_bands)

    return np.array(power_bands)


def extract_power_bands_over_time(data, sampling_rate=250, window_length=250, overlap=0.5, target_sampling_rate=250):
    """
    Compute power in EMG frequency bands over sliding windows.

    Args:
        data: Signal array (samples x channels)
        sampling_rate: Sampling rate in Hz
        window_length: Window length in samples
        overlap: Fractional overlap between windows
        target_sampling_rate: Target output sampling rate (for upsampling)

    Returns:
        Upsampled power-band array, or None on failure
    """
    try:
        power_bands_over_time = []

        frequency_bands = [(15, 20), (20, 25), (25, 30), (30, 35), (35, 45)]

        window_samples = int(window_length)
        overlap_samples = int(window_samples * overlap)
        num_windows = (len(data) - window_samples) // (window_samples - overlap_samples) + 1

        if num_windows <= 0:
            printError(f"[emgUtils] extract_power_bands_over_time: not enough data for even one window "
                       f"(data length={len(data)}, window={window_samples})")
            return None

        freqs, _ = welch(data[:, 0], fs=sampling_rate, nperseg=window_samples, noverlap=overlap_samples)

        for i in range(num_windows):
            start_idx = i * (window_samples - overlap_samples)
            end_idx = start_idx + window_samples
            windowed_data = data[start_idx:end_idx, :]

            window_power_bands = []
            for channel in range(data.shape[1]):
                channel_power_bands = []
                for band in frequency_bands:
                    _, psd = welch(windowed_data[:, channel], fs=sampling_rate,
                                   nperseg=window_samples, noverlap=overlap_samples)
                    lower_freq, upper_freq = band
                    band_indices = np.where((freqs >= lower_freq) & (freqs <= upper_freq))
                    band_power = np.trapz(psd[band_indices], dx=freqs[1])
                    channel_power_bands.append(band_power)
                window_power_bands.append(channel_power_bands)
            power_bands_over_time.append(window_power_bands)

        power_bands_over_time = np.array(power_bands_over_time)

        upsampling_factor = data.shape[0] // power_bands_over_time.shape[0]
        if upsampling_factor < 1:
            printError(f"[emgUtils] extract_power_bands_over_time: upsampling factor is < 1 "
                       f"(data={data.shape[0]}, windows={power_bands_over_time.shape[0]})")
            return None

        upsampler = interp1d(
            np.arange(power_bands_over_time.shape[0]),
            power_bands_over_time,
            axis=0,
            kind='linear',
        )
        upsampled = upsampler(
            np.linspace(0, power_bands_over_time.shape[0] - 1,
                        power_bands_over_time.shape[0] * upsampling_factor)
        )
        return upsampled

    except Exception as e:
        printError(f"[emgUtils] extract_power_bands_over_time failed: {e}")
        printError(f"[emgUtils] Traceback:\n{traceback.format_exc()}")
        return None


def computeSampEntropy(emgSignals, m=2):
    """
    Calculate the Sample Entropy of the signals.
    
    Args:
        emgSignals: EMG signal array
        m: Embedding dimension (default: 2)
        
    Returns:
        Sample entropy values for each channel
        
    Note: Requires EntropyHub package
    """
    if not ENTROPYHUB_AVAILABLE:
        printWarning("[emgUtils] Sample entropy computation skipped - EntropyHub not installed")
        return np.zeros((emgSignals.shape[1],))

    sampleEntropy = np.zeros((emgSignals.shape[1],))

    for i in range(emgSignals.shape[1]):
        tmp, _, _ = EH.SampEn(emgSignals[:, i], m=2)
        sampleEntropy[i] = tmp[2]

    return sampleEntropy


def loadHermesSignals(recPath, reReference=True):
    """Backward-compatible alias for :func:`loadEXG` in ``HermesDataInterface``."""
    from nucleuskit_pipeline.hermes.devices.HermesDataInterface import loadEXG

    return loadEXG(recPath, re_reference=reReference)
