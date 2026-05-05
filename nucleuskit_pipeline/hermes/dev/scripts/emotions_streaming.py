import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import json


# ============================================================
# Soft clip compressor
# ============================================================
def soft_clip(x, threshold):
    return threshold * np.tanh(x / threshold)


# ============================================================
# Online calibrator: direction + baseline+scale energy
# ============================================================
class OnlineCalibrator:
    def __init__(self,
                 num_channels,
                 alpha_mean=0.001,
                 alpha_var=0.003,
                 z_clip=3.5,
                 eps=1e-8,
                 energy_buffer_size=600,   # stores last N energy values (~5 min if RMS every 0.5s)
                 energy_clip=3.0,          # clip normalized energy before sigmoid
                 energy_gain=1.0):         # contrast for final sigmoid mapping

        self.num_channels = num_channels

        # Direction normalization
        self.alpha_mean = alpha_mean
        self.alpha_var = alpha_var
        self.z_clip = z_clip
        self.eps = eps
        self.initialized = False
        self.mean = np.zeros(num_channels)
        self.var = np.ones(num_channels)

        # Energy normalization (baseline+scale)
        self.energy_buffer_size = energy_buffer_size
        self.energy_buffer = []
        self.energy_clip = energy_clip
        self.energy_gain = energy_gain

    # ----------------------------
    def _init_from_sample(self, x):
        self.mean = x.copy()
        self.var = np.ones_like(x)
        self.initialized = True

    # ----------------------------
    def update_and_normalize(self, rms_vec):
        """
        Output: feature vector = [unit direction..., normalized energy (0–1)]
        """

        r = np.asarray(rms_vec).astype(float)
        x = np.log(r + self.eps)

        # --------- Direction (z-score) ---------
        if not self.initialized:
            self._init_from_sample(x)

        std = np.sqrt(self.var + self.eps)
        z_prov = (x - self.mean) / std

        # Outlier detection
        is_outlier = np.any(np.abs(z_prov) > self.z_clip)

        if not is_outlier:
            # Update mean
            self.mean = (1 - self.alpha_mean) * self.mean + self.alpha_mean * x

            # Update variance
            dev = x - self.mean
            self.var = (1 - self.alpha_var) * self.var + self.alpha_var * (dev**2)

            std = np.sqrt(self.var + self.eps)
            z = (x - self.mean) / std
        else:
            z = z_prov

        # Unit direction vector
        norm = np.linalg.norm(z) + self.eps
        direction = z / norm

        # --------- Energy normalization (baseline + scale) ---------
        # Raw amplitude energy
        energy_mag = np.sqrt(np.mean(r**2))

        # Rolling buffer update
        self.energy_buffer.append(energy_mag)
        if len(self.energy_buffer) > self.energy_buffer_size:
            self.energy_buffer.pop(0)

        # Default: energy_norm = 0.5 until enough history
        energy_norm = 0.5

        if len(self.energy_buffer) >= 30:  # enough samples to compute percentiles
            buf = np.sort(np.array(self.energy_buffer))
            n = len(buf)

            idx_2 = max(0, int(0.02 * n))
            idx_30 = max(idx_2 + 1, int(0.30 * n))
            idx_60 = max(idx_30 + 1, int(0.60 * n))

            # Neutral baseline = median between 2% and 30%
            neutral_baseline = np.median(buf[idx_2:idx_30])

            # Neutral scale = median between 30% and 60% minus baseline
            upper_neutral = np.median(buf[idx_30:idx_60])
            neutral_scale = max(upper_neutral - neutral_baseline, self.eps)

            # True normalized RMS energy (z-like)
            E_std = (energy_mag - neutral_baseline) / neutral_scale

            # Clip and map to 0–1
            E_std = np.clip(E_std, -self.energy_clip, self.energy_clip)
            shift = 1.5  # tune between 0.5 and 2.0
            energy_norm = 1.0 / (1.0 + np.exp(-self.energy_gain * (E_std - shift)))

        # --------- Final feature vector ---------
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
        self.zi = np.zeros((num_channels, max(len(self.a), len(self.b)) - 1))

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
    # Load EEG
    # --------------------------------------------------------
    data = np.loadtxt("rawEEG_0.csv", delimiter=",")
    signals = data[:, 1:]
    num_channels = signals.shape[1]

    processor = StreamProcessor(num_channels=num_channels)
    calibrator = OnlineCalibrator(num_channels=num_channels)

    filtered_stream = []
    rms_stream = []
    feature_stream = []

    # --------------------------------------------------------
    # Streaming loop
    # --------------------------------------------------------
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
    # RMS timestamps (0.5s)
    # --------------------------------------------------------
    rms_interval = 0.5
    rms_times = np.arange(len(rms_stream)) * rms_interval

    # --------------------------------------------------------
    # Load emotion events
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Plot RMS with emotion markers
    # --------------------------------------------------------
    plt.figure(figsize=(16, 6))

    rms_offset = np.max(rms_stream) * 1.3

    for ch in range(num_channels):
        plt.plot(rms_times, rms_stream[:, ch] + ch * rms_offset)

    for _, row in expr_df.iterrows():
        t = row["timestamp"]
        plt.axvline(t, color='red', linestyle='--', alpha=0.6)
        plt.text(t + 0.05,
                 rms_offset * (num_channels + 0.2),
                 row["emotion"],
                 rotation=90,
                 fontsize=10,
                 color='red')

    plt.title("Streaming RMS with Emotion Labels")
    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot Feature Stream (direction + normalized energy)
    # --------------------------------------------------------
    plt.figure(figsize=(16, 6))

    # Direction components
    for f in range(num_channels):
        plt.plot(rms_times, feature_stream[:, f], alpha=0.8)

    # Normalize energy (last element)
    plt.plot(rms_times, feature_stream[:, -1],
             color='black', linewidth=2, label="Energy (0–1)")

    for _, row in expr_df.iterrows():
        t = row["timestamp"]
        plt.axvline(t, color='red', linestyle='--', alpha=0.6)
        plt.text(t + 0.05,
                 1.15,
                 row["emotion"],
                 rotation=90,
                 fontsize=10,
                 color='red')

    plt.ylim(-1.2, 1.2)
    plt.title("Unit Direction + Baseline+Scale Energy (0–1) with Emotion Labels")
    plt.xlabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()
