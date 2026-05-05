import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi  # <<< CHANGED: added lfilter_zi
import json


# ============================================================
# Soft clip compressor
# ============================================================
def soft_clip(x, threshold):
    return threshold * np.tanh(x / threshold)


# ============================================================
# Compute offline energy parameters (baseline + scale)
# ============================================================
def compute_offline_energy_params(rms_stream, eps=1e-8):
    """
    rms_stream: shape (W, C)
    Returns:
        baseline, scale
    based on offline 2–30% and 30–60% percentiles
    """

    # Global window-level energy magnitude
    energy_mag = np.sqrt(np.mean(rms_stream**2, axis=1))  # shape (W,)

    buf = np.sort(energy_mag)
    n = len(buf)

    idx_2  = max(0,         int(0.02 * n))
    idx_30 = max(idx_2 + 1, int(0.30 * n))
    idx_60 = max(idx_30+1,  int(0.60 * n))

    baseline = np.median(buf[idx_2:idx_30])
    upper    = np.median(buf[idx_30:idx_60])

    scale = max(upper - baseline, eps)

    return baseline, scale


# ============================================================
# NEW: Compute offline direction parameters (mean + var in log-RMS)
#     with robust clipping so early weird data doesn't dominate.
# ============================================================
def compute_offline_direction_params(rms_stream,
                                     eps=1e-8,
                                     clip_percentiles=(1.0, 99.0)):
    """
    rms_stream: shape (W, C)
    Returns:
        mean_log, var_log

    We work in log-RMS space (same transform as used online)
    and clip extremes per channel to reduce the effect of outliers
    (including early calibration artifacts), but we still use
    *all* windows to compute stats.
    """
    r = np.asarray(rms_stream).astype(float)
    x = np.log(r + eps)  # shape (W, C)

    if clip_percentiles is not None:
        lo_p, hi_p = clip_percentiles
        # Per-channel clipping limits
        lo = np.percentile(x, lo_p, axis=0)
        hi = np.percentile(x, hi_p, axis=0)
        x = np.clip(x, lo, hi)

    mean_log = np.mean(x, axis=0)
    dev = x - mean_log
    var_log = np.mean(dev**2, axis=0)
    var_log = np.maximum(var_log, eps)

    return mean_log, var_log


# ============================================================
# Offline calibrator: fixed direction stats + offline energy
# ============================================================
class OfflineCalibrator:
    def __init__(self,
                 num_channels,
                 offline_mean_log,
                 offline_var_log,
                 offline_baseline,
                 offline_scale,
                 z_clip=3.5,
                 eps=1e-8,
                 energy_clip=3.0,
                 energy_gain=1.0,
                 shift=1.5):

        self.num_channels = num_channels

        # Fixed direction stats (log-RMS space)
        self.mean_log = np.asarray(offline_mean_log).astype(float)
        self.var_log  = np.asarray(offline_var_log).astype(float)
        assert self.mean_log.shape[0] == num_channels
        assert self.var_log.shape[0] == num_channels

        self.z_clip = z_clip
        self.eps = eps

        # Offline energy parameters
        self.offline_baseline = offline_baseline
        self.offline_scale    = offline_scale

        # Energy normalization
        self.energy_clip = energy_clip
        self.energy_gain = energy_gain
        self.shift       = shift

    # ----------------------------
    def update_and_normalize(self, rms_vec):
        """
        Output: feature vector = [unit direction..., normalized energy (0–1)]
        This is *offline* in the sense that it does not adapt;
        it just applies fixed mean/var + energy params to each RMS vector.
        """
        r = np.asarray(rms_vec).astype(float)
        x = np.log(r + self.eps)

        # --------- Direction (offline, no adaptation) ---------
        std_log = np.sqrt(self.var_log + self.eps)
        z = (x - self.mean_log) / std_log

        # Clip to avoid crazy extremes (still keeps all windows usable)
        z = np.clip(z, -self.z_clip, self.z_clip)

        # Unit direction vector
        direction = z / (np.linalg.norm(z) + self.eps)

        # --------- Energy normalization (offline parameters) ---------
        energy_mag = np.sqrt(np.mean(r**2))

        # Standardize using offline parameters
        E_std = (energy_mag - self.offline_baseline) / self.offline_scale

        # Clip and apply sigmoid
        E_std = np.clip(E_std, -self.energy_clip, self.energy_clip)
        energy_norm = 1.0 / (1.0 + np.exp(-self.energy_gain * (E_std - self.shift)))

        return np.concatenate([direction, [energy_norm]])


# ============================================================
# Streaming processor: filtering + soft clip + sliding RMS
# ============================================================
class StreamProcessor:
    def __init__(self,
                 fs=250,
                 low=15, high=45,
                 order=4,
                 rms_win=500,
                 rms_stride=125,
                 clip_threshold=80,
                 num_channels=8):

        self.fs = fs
        self.num_channels = num_channels

        # Band-pass filter
        self.b, self.a = butter(order,
                                [low/(fs/2), high/(fs/2)],
                                btype="band")

        # ### CHANGED: better initialization of filter state
        # Use lfilter_zi to approximate steady-state for zero input.
        zi_single = lfilter_zi(self.b, self.a)  # shape (N-1,)
        # For axis=0 filtering with shape (T, C), zi must be (N-1, C)
        self.zi = np.tile(zi_single[:, np.newaxis], (1, num_channels))

        # Soft clipping
        self.clip_threshold = clip_threshold

        # RMS params
        self.rms_win = rms_win
        self.rms_stride = rms_stride
        self.buffer = np.zeros((0, num_channels))
        self.samples_since_last_rms = 0

    def process_sample(self, sample):
        sample = np.asarray(sample).reshape(1, -1)

        # Filter
        y, self.zi = lfilter(self.b, self.a, sample, axis=0, zi=self.zi)

        # Soft clip
        y = soft_clip(y, self.clip_threshold)

        # RMS buffer
        self.buffer = np.vstack([self.buffer, y])
        self.samples_since_last_rms += 1

        if self.buffer.shape[0] >= self.rms_win and \
           self.samples_since_last_rms >= self.rms_stride:

            window = self.buffer[-self.rms_win:]
            rms_vec = np.sqrt(np.mean(window**2, axis=0))
            self.samples_since_last_rms = 0
            return y.flatten(), rms_vec

        return y.flatten(), None


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Load EEG offline and compute offline parameters
    # --------------------------------------------------------
    data = np.loadtxt("rawEEG_0.csv", delimiter=",")
    signals = data[:, 1:]
    num_channels = signals.shape[1]

    # Temporary offline RMS extraction (for parameter estimation)
    # We reuse the StreamProcessor logic but just collect RMS.
    processor_temp = StreamProcessor(num_channels=num_channels)

    all_rms = []
    for i in range(len(signals)):
        _, rms_vec = processor_temp.process_sample(signals[i])
        if rms_vec is not None:
            all_rms.append(rms_vec)

    all_rms = np.array(all_rms)

    # Compute offline energy baseline + scale
    offline_baseline, offline_scale = compute_offline_energy_params(all_rms)

    # NEW: compute offline direction stats in log-RMS space
    offline_mean_log, offline_var_log = compute_offline_direction_params(all_rms)

    # --------------------------------------------------------
    # 2. Now run the feature extraction using offline params
    # --------------------------------------------------------
    processor = StreamProcessor(num_channels=num_channels)
    calibrator = OfflineCalibrator(
        num_channels=num_channels,
        offline_mean_log=offline_mean_log,
        offline_var_log=offline_var_log,
        offline_baseline=offline_baseline,
        offline_scale=offline_scale,
        energy_gain=1.0,
        shift=1.5
    )

    filtered_stream = []
    rms_stream = []
    feature_stream = []

    for i in range(len(signals)):
        sample = signals[i]
        filt, rms_vec = processor.process_sample(sample)
        filtered_stream.append(filt)

        if rms_vec is not None:
            rms_stream.append(rms_vec)
            feature_stream.append(calibrator.update_and_normalize(rms_vec))

    filtered_stream = np.array(filtered_stream)
    rms_stream = np.array(rms_stream)
    feature_stream = np.array(feature_stream)

    # --------------------------------------------------------
    # Plotting logic unchanged…
    # --------------------------------------------------------

    # RMS timestamps (0.5s)   (rms_stride / fs = 125/250 = 0.5 s)
    rms_times = np.arange(len(rms_stream)) * 0.5

    # Load emotion labels
    event_df = pd.read_csv("events.csv")
    event_df["timestamp"] -= event_df["timestamp"].iloc[0]

    expr_df = event_df[event_df["tag"] == "EXPRESSION"].copy()

    def parse_json_field(s):
        try:
            return json.loads(s)
        except:
            return json.loads(s.replace('""', '"'))

    expr_df["emotion"] = expr_df["values"].apply(
        lambda v: parse_json_field(v)["emotion"]
    )

    # Plot RMS
    plt.figure(figsize=(16, 6))
    rms_offset = np.max(rms_stream) * 1.3

    for ch in range(num_channels):
        plt.plot(rms_times, rms_stream[:, ch] + ch*rms_offset)

    for _, row in expr_df.iterrows():
        plt.axvline(row["timestamp"], color='red', linestyle='--', alpha=0.6)
        plt.text(row["timestamp"] + 0.05,
                 rms_offset*(num_channels+0.2),
                 row["emotion"],
                 rotation=90, fontsize=10, color='red')

    plt.title("Streaming RMS with Emotion Labels")
    plt.tight_layout()
    plt.show()

    # Plot features
    plt.figure(figsize=(16, 6))

    # Direction features per channel
    for ch in range(num_channels):
        plt.plot(rms_times, feature_stream[:, ch], alpha=0.8)

    # Energy feature (last dimension)
    plt.plot(rms_times, feature_stream[:, -1], color='black', linewidth=2)

    for _, row in expr_df.iterrows():
        plt.axvline(row["timestamp"], color='red', linestyle='--', alpha=0.6)
        plt.text(row["timestamp"] + 0.05,
                 1.15,
                 row["emotion"],
                 rotation=90, fontsize=10, color='red')

    plt.ylim(-1.2, 1.2)
    plt.title("Direction + Offline-normalized Energy")
    plt.tight_layout()
    plt.show()
