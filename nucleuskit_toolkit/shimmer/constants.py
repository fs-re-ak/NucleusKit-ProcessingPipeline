"""Constants for the Shimmer3 wristband hardware."""


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
