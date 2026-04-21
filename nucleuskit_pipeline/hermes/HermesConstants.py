
class HermesConstants(object):

    SAMPLING_RATE = 250
    SAMPLING_PERIOD = (1/SAMPLING_RATE)

    CHANNELS = {"AF8": 0, "AF7": 1, "CHEEK_R": 2, "CHEEK_L": 3, "EAR_R": 4, "AFz": 5, "BROW_L": 6, "NOSE": 7}
    CHANNEL_NAMES = list(CHANNELS.keys())

    EMG_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]
    EEG_CHANNELS = [0, 1, 4, 5]

    POWER_BANDS = {"Delta": 0, "Theta": 1, "Alpha": 2, "Beta": 3, "Gamma": 4}
    NB_BANDS = len(POWER_BANDS)
    BANDS_DEFINITIONS = [[0, 4],
                         [4, 8],
                         [8, 13],
                         [13, 22],
                         [30, 50]]

    POWER_BANDS_DEFAULT_WINDOW = 5 * SAMPLING_RATE
    POWER_BANDS_DEFAULT_OVERLAP = 1 * SAMPLING_RATE


class ShimmerConstants(object):

    # tmp signal definition
    SAMPLING_RATE = 51.2
    GSR_INDEX = 4
    NYQUIST_FREQ = SAMPLING_RATE / 2

    # ppg conditioning
    FILTER_ORDER = 4  # applied twice
    AVG_FILTER_FC = 0.5

    # HRV/BPM
    THRESHOLD = 20
    HRV_WINDOW_WIDTH = 11  # used for smoothing
    BPM_WINDOW_WIDTH = 11  # used for smoothing used to be 101
    MIN_BPM = 40
    MAX_BPM = 150

    # gsr conditioning
    GSR_BANDWIDTH = [0.01, 5]
    TONIC_FC = 5.0
    GSR_THRESHOLD_TIGHT = 10

    # arousal conditioning
    FILTER_ORDER = 4
    FIRST_STAGE_BANDWIDTH = [0.03, 3]
    SMOOTHING_CUTOFF = 0.03

    THRESHOLD = 2
