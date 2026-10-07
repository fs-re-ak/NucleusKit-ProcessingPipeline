"""Handcrafted EMG feature extraction for classical_v1.

All features operate on the already-filtered 15–40 Hz window.
Input shape: ``[n_channels, n_samples]``.

Available feature sets
----------------------
``"full"`` (96 features)
    12 features per channel × 8 channels.
    Time-domain (7): RMS, MAV, WL, ZC, SSC, VAR, IEMG
    Frequency-domain (5): band-power, mean freq, median freq, freq ratio
      (low 15–25 Hz vs high 25–40 Hz), spectral entropy

``"rms_unit"`` (9 features)
    Per-channel RMS normalised to a unit vector (8 values) plus the mean
    RMS across all channels (1 value).  Captures activation magnitude and
    its spatial distribution while being invariant to overall scale.

``"rms_zscore"`` (8 features)
    Per-channel RMS z-score normalised **across all windows of the same
    subject**: ``(rms_ch - subject_mean_ch) / (subject_std_ch + eps)``.
    Captures relative channel differences after removing each subject's
    individual baseline.  Requires subject-level calibration at inference
    time via :meth:`ClassicalEmotionModel.set_subject_context`.

``"extended"`` (24 features)
    Two amplitude-sensitive blocks (subject z-score normalised) and three
    scale-free blocks (no subject normalisation):

    * RMS z-score — all 8 channels (8 features)
    * WL z-score  — 7 EMG channels, Temporal_EEG excluded (7 features)
    * MDF         — 7 EMG channels, 20–120 Hz band (7 features)
    * Asymmetry   — signed Zygomatic L/R ratio (1 feature)
    * Antagonist correlation — Glabella × Supraorbital_L envelope (1 feature)

``"extended_nm_a"`` (25 features)
    ``extended`` plus NM ratio A (Glabella vs Nasolabial).

``"extended_nm_a_lp"`` (25 features)
    Same layout as ``extended_nm_a``, but RMS and WL are normalised against
    a label-free low-power rest reference (5th–20th percentile of overall
    EMG power) instead of full-recording z-score.

"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(samples: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """Extract a flat feature vector from one EMG window.

    Parameters
    ----------
    samples:
        Shape ``[n_channels, n_samples]``, float32 or float64.
    sample_rate_hz:
        Sampling rate used for frequency-domain features.

    Returns
    -------
    np.ndarray
        1-D float64 feature vector of length ``n_channels * N_FEATURES_PER_CHANNEL``.
    """
    n_channels, n_samples = samples.shape
    per_channel = [
        _channel_features(samples[ch, :], sample_rate_hz)
        for ch in range(n_channels)
    ]
    return np.concatenate(per_channel).astype(np.float64)


N_FEATURES_PER_CHANNEL = 12


def feature_names(channel_names: list[str]) -> list[str]:
    """Return an ordered list of feature names matching ``extract_features`` output."""
    suffixes = [
        "rms", "mav", "wl", "zc", "ssc", "var", "iemg",
        "band_power", "mean_freq", "median_freq", "freq_ratio", "spectral_entropy",
    ]
    names: list[str] = []
    for ch in channel_names:
        for s in suffixes:
            names.append(f"{ch}_{s}")
    return names


# ---------------------------------------------------------------------------
# Per-channel feature computation
# ---------------------------------------------------------------------------


def _channel_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Compute all features for one channel."""
    td = _time_domain(x)
    fd = _freq_domain(x, fs)
    return np.concatenate([td, fd])


def _time_domain(x: np.ndarray) -> np.ndarray:
    """7 time-domain EMG features."""
    rms = np.sqrt(np.mean(x ** 2))
    mav = np.mean(np.abs(x))
    wl = np.sum(np.abs(np.diff(x)))  # waveform length
    zc = np.sum(
        ((x[:-1] * x[1:]) < 0) & (np.abs(np.diff(x)) > 1e-6)
    )  # zero crossings with threshold
    ssc = np.sum(
        (np.diff(x[:-1]) * np.diff(x[1:])) < 0
    )  # slope sign changes
    var = np.var(x)
    iemg = np.sum(np.abs(x))  # integrated EMG
    return np.array([rms, mav, wl, float(zc), float(ssc), var, iemg])


def _freq_domain(x: np.ndarray, fs: float) -> np.ndarray:
    """5 frequency-domain EMG features."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2

    # Band-power (15–40 Hz — the already-filtered band)
    band_mask = (freqs >= 15.0) & (freqs <= 40.0)
    band_power = np.sum(power[band_mask]) if band_mask.any() else 0.0

    # Mean frequency
    total_power = np.sum(power[band_mask]) if band_mask.any() else 1.0
    if total_power > 0 and band_mask.any():
        mean_freq = np.sum(freqs[band_mask] * power[band_mask]) / total_power
    else:
        mean_freq = 0.0

    # Median frequency: cumulative power crosses 50%
    if band_mask.any():
        cum_power = np.cumsum(power[band_mask])
        half = cum_power[-1] / 2.0
        idx_med = np.searchsorted(cum_power, half)
        band_freqs = freqs[band_mask]
        median_freq = float(band_freqs[min(idx_med, len(band_freqs) - 1)])
    else:
        median_freq = 0.0

    # Frequency ratio: low (15–25 Hz) vs high (25–40 Hz)
    low_mask = (freqs >= 15.0) & (freqs < 25.0)
    high_mask = (freqs >= 25.0) & (freqs <= 40.0)
    low_p = np.sum(power[low_mask]) if low_mask.any() else 0.0
    high_p = np.sum(power[high_mask]) if high_mask.any() else 0.0
    freq_ratio = low_p / (high_p + 1e-12)

    # Spectral entropy (within band)
    if band_mask.any():
        p_norm = power[band_mask] / (np.sum(power[band_mask]) + 1e-12)
        # Clip to avoid log(0)
        p_norm = np.clip(p_norm, 1e-12, None)
        sp_entropy = float(-np.sum(p_norm * np.log(p_norm)))
    else:
        sp_entropy = 0.0

    return np.array([band_power, mean_freq, median_freq, freq_ratio, sp_entropy])


# ---------------------------------------------------------------------------
# RMS unit-vector feature set (9 features)
# ---------------------------------------------------------------------------


def extract_rms_unit_features(
    samples: np.ndarray,
    sample_rate_hz: float = 250.0,  # accepted for interface uniformity, unused
) -> np.ndarray:
    """Extract a 9-feature RMS unit-vector representation.

    Parameters
    ----------
    samples:
        Shape ``[n_channels, n_samples]``.
    sample_rate_hz:
        Accepted for interface compatibility with ``extract_features``; not
        used by this extractor.

    Returns
    -------
    np.ndarray
        Float64 vector of length 9:
        ``[rms_unit_ch1, ..., rms_unit_ch8, mean_rms]``

        The first 8 values are the per-channel RMS values divided by their
        L2 norm so the vector lies on the unit sphere.  The 9th value is
        the unscaled mean RMS across channels, which captures overall
        activation magnitude that the normalisation removes.
    """
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2, axis=1))  # [n_channels]
    norm = np.linalg.norm(rms)
    rms_unit = rms / (norm + 1e-12)   # unit vector; safe when signal is silent
    mean_rms = float(np.mean(rms))
    return np.concatenate([rms_unit, [mean_rms]])


# ---------------------------------------------------------------------------
# Subject-level z-score helpers  (rms_zscore feature set)
# ---------------------------------------------------------------------------


def compute_rms_batch(X: np.ndarray) -> np.ndarray:
    """Per-channel RMS for a batch of windows.

    Parameters
    ----------
    X:
        Shape ``[n_windows, n_channels, n_samples]``.

    Returns
    -------
    np.ndarray
        Shape ``[n_windows, n_channels]``, float64.
    """
    return np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=2))


def compute_subject_zscore_batch(
    X: np.ndarray,
    participant_ids: list,
) -> np.ndarray:
    """Per-channel RMS z-score normalised per subject across that subject's windows.

    For each subject independently:

    1. Compute RMS per channel for every window → ``[N_subj, n_channels]``
    2. Compute per-channel *mean* and *std* across that subject's windows
    3. Return ``(rms - mean) / (std + eps)``

    Parameters
    ----------
    X:
        Shape ``[n_windows, n_channels, n_samples]``.
    participant_ids:
        Subject ID per window, same length as ``X``.

    Returns
    -------
    np.ndarray
        Shape ``[n_windows, n_channels]``, float64 — subject-normalised features.
    """
    rms    = compute_rms_batch(X)          # [n_windows, n_channels]
    result = np.zeros_like(rms)
    pids   = np.array(participant_ids)

    for subj in np.unique(pids):
        mask       = pids == subj
        subj_rms   = rms[mask]             # [N_subj, n_channels]
        mean        = subj_rms.mean(axis=0)
        std         = subj_rms.std(axis=0)
        result[mask] = (subj_rms - mean) / (std + 1e-12)

    return result


def extract_rms_zscore_features(
    samples: np.ndarray,
    sample_rate_hz: float = 250.0,
) -> np.ndarray:
    """Raw per-channel RMS for a single window — pass-through for rms_zscore.

    The subject z-score normalisation is applied at the dataset level by
    :func:`compute_subject_zscore_batch` (training) or by calling
    :meth:`ClassicalEmotionModel.set_subject_context` before inference.
    This function is kept in :data:`FEATURE_SETS` so the model can identify
    its feature set; it is **not** called directly during training or
    inference for ``rms_zscore`` models.

    Parameters
    ----------
    samples:
        Shape ``[n_channels, n_samples]``.
    sample_rate_hz:
        Accepted for interface compatibility; not used.

    Returns
    -------
    np.ndarray
        Float64 raw RMS vector of length ``n_channels``.
    """
    return np.sqrt(np.mean(samples.astype(np.float64) ** 2, axis=1))


# ---------------------------------------------------------------------------
# Extended feature set helpers
# ---------------------------------------------------------------------------

# Temporal_EEG is not a muscle signal; exclude it from WL and MDF.
_TEMPORAL_EEG_NAME = "Temporal_EEG"

# Standard channel layout (0-based) for the default 8-channel montage.
# Used only as a fallback; production code passes indices explicitly.
_DEFAULT_EMG_INDICES = [0, 1, 2, 3, 5, 6, 7]  # all channels except idx 4
_DEFAULT_ZYG_L = 2   # Zygomatic_L
_DEFAULT_ZYG_R = 3   # Zygomatic_R
_DEFAULT_ANT_A = 5   # Glabella (corrugator territory)
_DEFAULT_ANT_B = 0   # Supraorbital_L (frontalis territory)
_DEFAULT_ANT_C = 1   # Supraorbital_R (frontalis, right)
_DEFAULT_ANT_D = 6   # Temporal_R (jaw / temporalis territory)

_ENV_MA_LEN = 25     # 100 ms moving-average for envelope at 250 Hz
_MDF_BAND    = (20.0, 120.0)  # Hz — safe below Nyquist at 250 Hz


def compute_wl_batch(X: np.ndarray) -> np.ndarray:
    """Waveform length per channel for a batch of windows.

    Parameters
    ----------
    X:
        Shape ``[n_windows, n_channels, n_samples]``.

    Returns
    -------
    np.ndarray
        Shape ``[n_windows, n_channels]``, float64.
    """
    X = X.astype(np.float64)
    return np.sum(np.abs(np.diff(X, axis=2)), axis=2)


def _compute_mdf_channel(x: np.ndarray, fs: float, band: tuple = _MDF_BAND) -> float:
    """Median frequency for a single channel window.

    Removes DC, applies Hann taper, computes FFT periodogram, restricts to
    ``band`` Hz, then finds the 50% cumulative-power crossing with linear
    interpolation.
    """
    x = x.astype(np.float64)
    x = x - x.mean()                        # remove DC offset
    hann = np.hanning(len(x))
    x = x * hann                             # reduce spectral leakage
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2

    lo, hi = band
    band_mask = (freqs >= lo) & (freqs <= hi)
    if not band_mask.any():
        return float((lo + hi) / 2.0)

    psd   = power[band_mask]
    fband = freqs[band_mask]
    cum   = np.cumsum(psd)
    half  = cum[-1] / 2.0

    idx = int(np.searchsorted(cum, half))
    idx = min(idx, len(cum) - 1)

    if idx == 0 or cum[idx] == cum[idx - 1]:
        return float(fband[idx])

    # Linear interpolation between bracketing bins
    t = (half - cum[idx - 1]) / (cum[idx] - cum[idx - 1])
    return float(fband[idx - 1] * (1.0 - t) + fband[idx] * t)


def compute_mdf_batch(
    X: np.ndarray,
    emg_indices: list,
    fs: float = 250.0,
) -> np.ndarray:
    """MDF for EMG-only channels across a batch of windows.

    Parameters
    ----------
    X:
        Shape ``[n_windows, n_channels, n_samples]`` — full channel set.
    emg_indices:
        Indices of EMG channels to include (Temporal_EEG excluded).
    fs:
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Shape ``[n_windows, len(emg_indices)]``, float64.
    """
    n_win = X.shape[0]
    n_emg = len(emg_indices)
    out = np.zeros((n_win, n_emg), dtype=np.float64)
    for w in range(n_win):
        for j, ch in enumerate(emg_indices):
            out[w, j] = _compute_mdf_channel(X[w, ch, :], fs)
    return out


def _envelope(x: np.ndarray, ma_len: int = _ENV_MA_LEN) -> np.ndarray:
    """Rectified moving-average envelope of a 1-D signal."""
    env = np.abs(x.astype(np.float64))
    kernel = np.ones(ma_len) / ma_len
    return np.convolve(env, kernel, mode="same")


def compute_asymmetry_batch(
    X: np.ndarray,
    zyg_l_idx: int = _DEFAULT_ZYG_L,
    zyg_r_idx: int = _DEFAULT_ZYG_R,
) -> np.ndarray:
    """Signed zygomaticus L/R asymmetry for a batch of windows.

    ``asym = (L_rms - R_rms) / (L_rms + R_rms + eps)``

    The result is bounded in [-1, +1].  Positive values indicate left
    dominance (CONTEMPT_LEFT pattern), negative values right dominance
    (CONTEMPT_RIGHT).

    Parameters
    ----------
    X:
        Shape ``[n_windows, n_channels, n_samples]``.

    Returns
    -------
    np.ndarray
        Shape ``[n_windows, 1]``, float64.
    """
    X = X.astype(np.float64)
    rms_l = np.sqrt(np.mean(X[:, zyg_l_idx, :] ** 2, axis=1))  # [n_windows]
    rms_r = np.sqrt(np.mean(X[:, zyg_r_idx, :] ** 2, axis=1))
    asym  = (rms_l - rms_r) / (rms_l + rms_r + 1e-6)
    return asym[:, np.newaxis]


def compute_antagonist_corr_batch(
    X: np.ndarray,
    ch_a_idx: int = _DEFAULT_ANT_A,
    ch_b_idx: int = _DEFAULT_ANT_B,
) -> np.ndarray:
    """Envelope Pearson correlation between an antagonist channel pair.

    Correlates the rectified moving-average envelopes.  Returns 0.0 when
    either channel is silent (RMS < 1e-6) to avoid noise-dominated values
    at rest.

    Parameters
    ----------
    X:
        Shape ``[n_windows, n_channels, n_samples]``.
    ch_a_idx:
        Index of the first channel (default: Glabella).
    ch_b_idx:
        Index of the second channel (default: Supraorbital_L).

    Returns
    -------
    np.ndarray
        Shape ``[n_windows, 1]``, float64, values in [-1, +1].
    """
    n_win = X.shape[0]
    out = np.zeros((n_win, 1), dtype=np.float64)
    for w in range(n_win):
        a = X[w, ch_a_idx, :].astype(np.float64)
        b = X[w, ch_b_idx, :].astype(np.float64)
        rms_a = float(np.sqrt(np.mean(a ** 2)))
        rms_b = float(np.sqrt(np.mean(b ** 2)))
        if rms_a < 1e-6 and rms_b < 1e-6:
            out[w, 0] = 0.0
            continue
        env_a = _envelope(a)
        env_b = _envelope(b)
        std_a = env_a.std()
        std_b = env_b.std()
        if std_a < 1e-12 or std_b < 1e-12:
            out[w, 0] = 0.0
            continue
        out[w, 0] = float(np.corrcoef(env_a, env_b)[0, 1])
    return out


def _subject_normalize_batch(M: np.ndarray, participant_ids: list) -> np.ndarray:
    """Per-subject z-score normalisation of a feature matrix.

    Parameters
    ----------
    M:
        Shape ``[n_windows, n_features]``, float64.
    participant_ids:
        Subject ID per window.

    Returns
    -------
    np.ndarray
        Same shape, normalised per subject.
    """
    result = np.zeros_like(M)
    pids   = np.array(participant_ids)
    for subj in np.unique(pids):
        mask = pids == subj
        m    = M[mask]
        result[mask] = (m - m.mean(axis=0)) / (m.std(axis=0) + 1e-12)
    return result


# ---------------------------------------------------------------------------
# Low-power (rest) reference normalisation
# ---------------------------------------------------------------------------

LOW_POWER_P_LO = 5.0    # reject below this percentile (dropouts / dead silence)
LOW_POWER_P_HI = 20.0   # cap of the rest / neutral band
LOW_POWER_MIN_N = 3     # minimum reference windows
# If p20/p50 is this high, the recording has little rest contrast.
LOW_POWER_REST_RATIO_MAX = 0.85
LOW_POWER_STD_FLOOR_FRAC = 0.05


def overall_emg_power(
    rms: np.ndarray,
    emg_indices: list = _DEFAULT_EMG_INDICES,
) -> np.ndarray:
    """Per-window mean RMS across EMG channels (Temporal_EEG excluded)."""
    return np.mean(rms[:, list(emg_indices)], axis=1)


def select_low_power_mask(
    power: np.ndarray,
    p_lo: float = LOW_POWER_P_LO,
    p_hi: float = LOW_POWER_P_HI,
    min_n: int = LOW_POWER_MIN_N,
) -> np.ndarray:
    """Boolean mask of the shared 5th–20th percentile power band.

    One mask for the whole recording — not per channel.
    """
    power = np.asarray(power, dtype=np.float64)
    n = power.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    lo, hi = np.percentile(power, [p_lo, p_hi])
    mask = (power >= lo) & (power <= hi)
    if int(mask.sum()) < min_n:
        mask = power <= hi
    if int(mask.sum()) < min_n:
        k = min(min_n, n)
        order = np.argsort(power)
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
    return mask


def rest_reference_ok(power: np.ndarray) -> bool:
    """True when the low-power band is clearly quieter than the median."""
    power = np.asarray(power, dtype=np.float64)
    if power.size < LOW_POWER_MIN_N:
        return False
    p20, p50 = np.percentile(power, [LOW_POWER_P_HI, 50.0])
    if p50 <= 1e-12:
        return False
    return bool((p20 / p50) < LOW_POWER_REST_RATIO_MAX)


def fit_low_power_stats(
    M: np.ndarray,
    power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """Per-channel mean/std on the shared low-power window set.

    Returns
    -------
    mean, std, ok, n_ref
    """
    mask = select_low_power_mask(power)
    ref = M[mask]
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    std = np.maximum(std, LOW_POWER_STD_FLOOR_FRAC * np.maximum(np.abs(mean), 1e-12))
    std = np.maximum(std, 1e-12)
    return mean, std, rest_reference_ok(power), int(mask.sum())


def _low_power_normalize_pair(
    rms: np.ndarray,
    wl: np.ndarray,
    participant_ids: list,
    emg_indices: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize RMS and WL with one shared low-power window set per subject."""
    power = overall_emg_power(rms, emg_indices)
    rms_out = np.zeros_like(rms)
    wl_out = np.zeros_like(wl)
    pids = np.array(participant_ids)
    for subj in np.unique(pids):
        sm = pids == subj
        pwr = power[sm]
        rms_mean, rms_std, ok, n_ref = fit_low_power_stats(rms[sm], pwr)
        wl_mean, wl_std, _, _ = fit_low_power_stats(wl[sm], pwr)
        if not ok:
            print(
                f"[low-power] {subj}: rest band may not be rest "
                f"(p20/p50 >= {LOW_POWER_REST_RATIO_MAX}, n_ref={n_ref})"
            )
        rms_out[sm] = (rms[sm] - rms_mean) / rms_std
        wl_out[sm] = (wl[sm] - wl_mean) / wl_std
    return rms_out, wl_out


def compute_extended_batch(
    X: np.ndarray,
    participant_ids: list,
    emg_indices: list = _DEFAULT_EMG_INDICES,
    zyg_l_idx: int    = _DEFAULT_ZYG_L,
    zyg_r_idx: int    = _DEFAULT_ZYG_R,
    ant_a_idx: int    = _DEFAULT_ANT_A,
    ant_b_idx: int    = _DEFAULT_ANT_B,
    fs: float         = 250.0,
    amplitude_norm: str = "zscore",
) -> np.ndarray:
    """Assemble the 24-feature extended vector for a batch of windows.

    Feature layout (24 total):

    ======  =====  =======================================================
    Offset  Count  Description
    ======  =====  =======================================================
    0       8      RMS (all channels, amplitude-normalised)
    8       7      WL  (EMG channels, amplitude-normalised)
    15      7      MDF         (EMG channels, raw Hz, 20–120 Hz band)
    22      1      Asymmetry   (Zygomatic L/R signed ratio, scale-free)
    23      1      Antagonist corr (Glabella × Supraorbital_L, scale-free)
    ======  =====  =======================================================

    ``amplitude_norm`` is ``"zscore"`` (full-recording per-subject mean/std)
    or ``"low_power"`` (mean/std of the 5th–20th percentile overall-power
    windows — a label-free rest reference).
    """
    # Amplitude-sensitive blocks
    rms_raw = compute_rms_batch(X)                               # [n, 8]
    wl_raw  = compute_wl_batch(X[:, emg_indices, :])             # [n, 7]

    if amplitude_norm == "low_power":
        rms_norm, wl_norm = _low_power_normalize_pair(
            rms_raw, wl_raw, participant_ids, emg_indices,
        )
    elif amplitude_norm == "zscore":
        rms_norm = _subject_normalize_batch(rms_raw, participant_ids)
        wl_norm  = _subject_normalize_batch(wl_raw,  participant_ids)
    else:
        raise ValueError(
            f"Unknown amplitude_norm '{amplitude_norm}'. "
            "Choose from: zscore, low_power"
        )

    # Scale-free blocks
    mdf  = compute_mdf_batch(X, emg_indices, fs)                 # [n, 7]
    asym = compute_asymmetry_batch(X, zyg_l_idx, zyg_r_idx)     # [n, 1]
    corr = compute_antagonist_corr_batch(X, ant_a_idx, ant_b_idx)  # [n, 1]

    return np.concatenate([rms_norm, wl_norm, mdf, asym, corr], axis=1)


def extract_extended_features(
    samples: np.ndarray,
    sample_rate_hz: float = 250.0,
) -> np.ndarray:
    """Pass-through stub for the extended feature set (single-window interface).

    Like ``extract_rms_zscore_features``, the extended set cannot be fully
    computed per-window in isolation — WL and RMS require subject-level
    calibration for z-scoring, and the batch functions are used during
    training and inference.  This stub exists solely so the feature set is
    registered in :data:`FEATURE_SETS`.
    """
    raise NotImplementedError(
        "extended features are assembled via compute_extended_batch (training) "
        "or ClassicalEmotionModel.predict (inference).  Do not call this stub directly."
    )


# ---------------------------------------------------------------------------
# NM ratio helpers (nose-to-mouth, scale-free, anger↔disgust targeted)
# ---------------------------------------------------------------------------

# Variant A: N = Glabella (corrugator / brow-bridge), M = Nasolabial (levator labii)
_DEFAULT_NM_A_N = 5   # Glabella
_DEFAULT_NM_A_M = [7]  # Nasolabial

# Variant B: N = Nasolabial (nasolabial fold), M = mean(Zygomatic_L, Zygomatic_R)
_DEFAULT_NM_B_N = 7         # Nasolabial
_DEFAULT_NM_B_M = [2, 3]    # Zygomatic_L, Zygomatic_R


def compute_nm_ratio_batch(
    X: np.ndarray,
    n_idx: int,
    m_indices: list,
    rest_eps: float = 1e-6,
) -> np.ndarray:
    """Signed nose-to-mouth RMS ratio for a batch of windows.

    ``ratio = (rms_N - rms_M) / (rms_N + rms_M + rest_eps)``

    When both channels are silent ``(rms_N + rms_M) < rest_eps``, the ratio
    is gated to 0.0 to avoid noise-dominated values at rest.

    Scale-free and bounded in [-1, +1].  Not subject-normalised.

    Parameters
    ----------
    X:
        Shape ``[n_windows, n_channels, n_samples]``.
    n_idx:
        Channel index for the "nose" region.
    m_indices:
        One or more channel indices for the "mouth" region; averaged if multiple.
    rest_eps:
        Rest-gate threshold: ratio is set to 0.0 when ``(rms_N + rms_M) < rest_eps``.

    Returns
    -------
    np.ndarray
        Shape ``[n_windows, 1]``, float64.
    """
    X = X.astype(np.float64)
    rms_n = np.sqrt(np.mean(X[:, n_idx, :] ** 2, axis=1))         # [n]
    rms_m_stack = np.stack(
        [np.sqrt(np.mean(X[:, i, :] ** 2, axis=1)) for i in m_indices],
        axis=1,
    )                                                               # [n, len(m_indices)]
    rms_m = rms_m_stack.mean(axis=1)                               # [n]

    denom  = rms_n + rms_m + rest_eps
    ratio  = (rms_n - rms_m) / denom
    # Gate to 0.0 where both channels are effectively silent
    silent = (rms_n + rms_m) < rest_eps
    ratio[silent] = 0.0
    return ratio[:, np.newaxis]


def compute_extended_nm_a_batch(
    X: np.ndarray,
    participant_ids: list,
    emg_indices: list = _DEFAULT_EMG_INDICES,
    zyg_l_idx: int    = _DEFAULT_ZYG_L,
    zyg_r_idx: int    = _DEFAULT_ZYG_R,
    ant_a_idx: int    = _DEFAULT_ANT_A,
    ant_b_idx: int    = _DEFAULT_ANT_B,
    nm_n_idx: int     = _DEFAULT_NM_A_N,
    nm_m_indices: list = _DEFAULT_NM_A_M,
    fs: float         = 250.0,
) -> np.ndarray:
    """25-feature extended_nm_a vector: extended (24) + Glabella/Nasolabial ratio (1).

    Feature layout (25 total):

    ======  =====  =======================================================
    Offset  Count  Description
    ======  =====  =======================================================
    0       24     All features from compute_extended_batch
    24      1      NM ratio A: (Glabella − Nasolabial) / (Glabella + Nasolabial + ε)
    ======  =====  =======================================================
    """
    base = compute_extended_batch(
        X, participant_ids,
        emg_indices=emg_indices,
        zyg_l_idx=zyg_l_idx, zyg_r_idx=zyg_r_idx,
        ant_a_idx=ant_a_idx, ant_b_idx=ant_b_idx,
        fs=fs,
    )
    nm = compute_nm_ratio_batch(X, nm_n_idx, nm_m_indices)
    return np.concatenate([base, nm], axis=1)


def compute_extended_nm_b_batch(
    X: np.ndarray,
    participant_ids: list,
    emg_indices: list = _DEFAULT_EMG_INDICES,
    zyg_l_idx: int    = _DEFAULT_ZYG_L,
    zyg_r_idx: int    = _DEFAULT_ZYG_R,
    ant_a_idx: int    = _DEFAULT_ANT_A,
    ant_b_idx: int    = _DEFAULT_ANT_B,
    nm_n_idx: int     = _DEFAULT_NM_B_N,
    nm_m_indices: list = _DEFAULT_NM_B_M,
    fs: float         = 250.0,
) -> np.ndarray:
    """25-feature extended_nm_b vector: extended (24) + Nasolabial/ZygMean ratio (1).

    Feature layout (25 total):

    ======  =====  =======================================================
    Offset  Count  Description
    ======  =====  =======================================================
    0       24     All features from compute_extended_batch
    24      1      NM ratio B: (Nasolabial − mean(Zyg_L, Zyg_R)) / (Nasolabial + mean(Zyg_L, Zyg_R) + ε)
    ======  =====  =======================================================
    """
    base = compute_extended_batch(
        X, participant_ids,
        emg_indices=emg_indices,
        zyg_l_idx=zyg_l_idx, zyg_r_idx=zyg_r_idx,
        ant_a_idx=ant_a_idx, ant_b_idx=ant_b_idx,
        fs=fs,
    )
    nm = compute_nm_ratio_batch(X, nm_n_idx, nm_m_indices)
    return np.concatenate([base, nm], axis=1)


def extract_extended_nm_a_features(samples: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """Stub — see compute_extended_nm_a_batch."""
    raise NotImplementedError(
        "extended_nm_a features are assembled via compute_extended_nm_a_batch (training) "
        "or ClassicalEmotionModel.predict (inference).  Do not call this stub directly."
    )


def compute_extended_nm_a_lp_batch(
    X: np.ndarray,
    participant_ids: list,
    emg_indices: list = _DEFAULT_EMG_INDICES,
    zyg_l_idx: int    = _DEFAULT_ZYG_L,
    zyg_r_idx: int    = _DEFAULT_ZYG_R,
    ant_a_idx: int    = _DEFAULT_ANT_A,
    ant_b_idx: int    = _DEFAULT_ANT_B,
    nm_n_idx: int     = _DEFAULT_NM_A_N,
    nm_m_indices: list = _DEFAULT_NM_A_M,
    fs: float         = 250.0,
) -> np.ndarray:
    """Same 25 features as extended_nm_a; RMS/WL use a low-power rest reference."""
    base = compute_extended_batch(
        X, participant_ids,
        emg_indices=emg_indices,
        zyg_l_idx=zyg_l_idx, zyg_r_idx=zyg_r_idx,
        ant_a_idx=ant_a_idx, ant_b_idx=ant_b_idx,
        fs=fs,
        amplitude_norm="low_power",
    )
    nm = compute_nm_ratio_batch(X, nm_n_idx, nm_m_indices)
    return np.concatenate([base, nm], axis=1)


def extract_extended_nm_a_lp_features(samples: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """Stub — see compute_extended_nm_a_lp_batch."""
    raise NotImplementedError(
        "extended_nm_a_lp features are assembled via compute_extended_nm_a_lp_batch "
        "(training) or ClassicalEmotionModel.predict (inference)."
    )


def extract_extended_nm_b_features(samples: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """Stub — see compute_extended_nm_b_batch."""
    raise NotImplementedError(
        "extended_nm_b features are assembled via compute_extended_nm_b_batch (training) "
        "or ClassicalEmotionModel.predict (inference).  Do not call this stub directly."
    )


def compute_extended_nm_ab_batch(
    X: np.ndarray,
    participant_ids: list,
    emg_indices: list = _DEFAULT_EMG_INDICES,
    zyg_l_idx: int    = _DEFAULT_ZYG_L,
    zyg_r_idx: int    = _DEFAULT_ZYG_R,
    ant_a_idx: int    = _DEFAULT_ANT_A,
    ant_b_idx: int    = _DEFAULT_ANT_B,
    fs: float         = 250.0,
) -> np.ndarray:
    """26-feature vector: extended (24) + NM ratio A + NM ratio B.

    Feature layout (26 total):

    ======  =====  =======================================================
    Offset  Count  Description
    ======  =====  =======================================================
    0       24     All features from compute_extended_batch
    24      1      NM ratio A: (Glabella − Nasolabial) / (Glabella + Nasolabial + ε)
    25      1      NM ratio B: (Nasolabial − mean(Zyg_L, Zyg_R)) / (Nasolabial + mean(Zyg) + ε)
    ======  =====  =======================================================
    """
    base = compute_extended_batch(
        X, participant_ids,
        emg_indices=emg_indices,
        zyg_l_idx=zyg_l_idx, zyg_r_idx=zyg_r_idx,
        ant_a_idx=ant_a_idx, ant_b_idx=ant_b_idx,
        fs=fs,
    )
    nm_a = compute_nm_ratio_batch(X, _DEFAULT_NM_A_N, _DEFAULT_NM_A_M)
    nm_b = compute_nm_ratio_batch(X, _DEFAULT_NM_B_N, _DEFAULT_NM_B_M)
    return np.concatenate([base, nm_a, nm_b], axis=1)


def extract_extended_nm_ab_features(samples: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """Stub — see compute_extended_nm_ab_batch."""
    raise NotImplementedError(
        "extended_nm_ab features are assembled via compute_extended_nm_ab_batch (training) "
        "or ClassicalEmotionModel.predict (inference).  Do not call this stub directly."
    )


def compute_extended_nm_a_corr_batch(
    X: np.ndarray,
    participant_ids: list,
    emg_indices: list = _DEFAULT_EMG_INDICES,
    zyg_l_idx: int    = _DEFAULT_ZYG_L,
    zyg_r_idx: int    = _DEFAULT_ZYG_R,
    ant_a_idx: int    = _DEFAULT_ANT_A,
    ant_b_idx: int    = _DEFAULT_ANT_B,
    ant_c_idx: int    = _DEFAULT_ANT_C,
    ant_d_idx: int    = _DEFAULT_ANT_D,
    nm_n_idx: int     = _DEFAULT_NM_A_N,
    nm_m_indices: list = _DEFAULT_NM_A_M,
    fs: float         = 250.0,
) -> np.ndarray:
    """27-feature vector: extended_nm_a (25) + two extra Glabella antagonist corrs.

    Feature layout (27 total):

    ======  =====  =======================================================
    Offset  Count  Description
    ======  =====  =======================================================
    0       25     All features from compute_extended_nm_a_batch
    25      1      Envelope corr: Glabella × Supraorbital_R
    26      1      Envelope corr: Glabella × Temporal_R
    ======  =====  =======================================================
    """
    base = compute_extended_nm_a_batch(
        X, participant_ids,
        emg_indices=emg_indices,
        zyg_l_idx=zyg_l_idx, zyg_r_idx=zyg_r_idx,
        ant_a_idx=ant_a_idx, ant_b_idx=ant_b_idx,
        nm_n_idx=nm_n_idx, nm_m_indices=nm_m_indices,
        fs=fs,
    )
    corr_sr = compute_antagonist_corr_batch(X, ant_a_idx, ant_c_idx)
    corr_tr = compute_antagonist_corr_batch(X, ant_a_idx, ant_d_idx)
    return np.concatenate([base, corr_sr, corr_tr], axis=1)


def extract_extended_nm_a_corr_features(samples: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """Stub — see compute_extended_nm_a_corr_batch."""
    raise NotImplementedError(
        "extended_nm_a_corr features are assembled via compute_extended_nm_a_corr_batch "
        "(training) or ClassicalEmotionModel.predict (inference).  Do not call this stub directly."
    )


# ---------------------------------------------------------------------------
# Feature set registry
# ---------------------------------------------------------------------------

FEATURE_SETS: dict[str, callable] = {
    "full":               extract_features,                    # 96
    "rms_unit":           extract_rms_unit_features,           # 9
    "rms_zscore":         extract_rms_zscore_features,         # 8
    "extended":           extract_extended_features,           # 24
    "extended_nm_a":      extract_extended_nm_a_features,      # 25
    "extended_nm_a_lp":   extract_extended_nm_a_lp_features,   # 25
    "extended_nm_b":      extract_extended_nm_b_features,      # 25
    "extended_nm_ab":     extract_extended_nm_ab_features,     # 26
    "extended_nm_a_corr": extract_extended_nm_a_corr_features, # 27
}

N_FEATURES = {
    "full":               96,
    "rms_unit":           9,
    "rms_zscore":         8,
    "extended":           24,
    "extended_nm_a":      25,
    "extended_nm_a_lp":   25,
    "extended_nm_b":      25,
    "extended_nm_ab":     26,
    "extended_nm_a_corr": 27,
}
