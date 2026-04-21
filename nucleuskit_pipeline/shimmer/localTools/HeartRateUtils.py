import sys
import os
import numpy as np
import pandas as pd
import csv
import scipy.signal as signal

from nucleuskit_pipeline.hermes.HermesConstants import ShimmerConstants
from nucleuskit_pipeline.shared.SignalProcessingUtils import smooth
from scipy.signal import butter, filtfilt, medfilt

from nucleuskit_pipeline.logging_utils import printError, printInfo


POINT_5_SECONDS_IN_SAMPLES = int(np.ceil(0.5 * ShimmerConstants.SAMPLING_RATE))

def normalizePPG(ppgSignal):

    normalizedSignal = np.zeros(ppgSignal.shape)

    # manage the first 3.5 seconds, using the first 7 seconds window as frame of reference
    gsrMin ,gsrMax = _computeRef(ppgSignal[:(2 * POINT_5_SECONDS_IN_SAMPLES)])
    normalizedSignal[:POINT_5_SECONDS_IN_SAMPLES] = (ppgSignal[:POINT_5_SECONDS_IN_SAMPLES] - gsrMin) / (gsrMax - gsrMin)

    # calculate each value, using the +/- 3.5 seconds of recording_data as reference
    for i in range(POINT_5_SECONDS_IN_SAMPLES , ppgSignal.shape[0] - POINT_5_SECONDS_IN_SAMPLES):
        gsrMin ,gsrMax = _computeRef(ppgSignal[(i - POINT_5_SECONDS_IN_SAMPLES):(i + POINT_5_SECONDS_IN_SAMPLES)])
        try:
            normalizedSignal[i] = (ppgSignal[i] - gsrMin) / (gsrMax - gsrMin)
        except:
            normalizedSignal[i] = 0

    # manage the last 3.5 seconds, using the last 7 seconds window as frame of reference
    gsrMin ,gsrMax = _computeRef(ppgSignal[-(2 * POINT_5_SECONDS_IN_SAMPLES):])
    normalizedSignal[-POINT_5_SECONDS_IN_SAMPLES:] = (ppgSignal[-POINT_5_SECONDS_IN_SAMPLES:] - gsrMin) / \
                                                   (gsrMax-gsrMin)

    return normalizedSignal

def _computeRef(refWindow):
    return np.min(refWindow), np.max(refWindow)


"""
Average filter, high-pass, with low fc
"""

def _avgFilt(ppgRec):

    # high-pass filter
    [b, a] = butter(4, ShimmerConstants.AVG_FILTER_FC/ShimmerConstants.NYQUIST_FREQ, btype='highpass')
    ppgRec = filtfilt(b, a, ppgRec)
    ppgRec = filtfilt(b, a, ppgRec)

    return ppgRec


def _singlePulseFilter(ppgRec):

    #copy
    ppgRec[ppgRec < 0] = 0

    # apply an average filter (highpass)
    pulseLogic = _avgFilt(ppgRec)

    # remove everything negative
    pulseLogic[pulseLogic < 0] = 0

    # apply an average filter AGAIN (highpass)
    pulseLogic = _avgFilt(pulseLogic)

    # apply threshold
    pulseLogic[pulseLogic < ShimmerConstants.THRESHOLD] = 0
    pulseLogic[pulseLogic >= ShimmerConstants.THRESHOLD] = 1

    # detect edges
    pulseLogic = np.diff(pulseLogic)
    # remove falling edge
    pulseLogic[pulseLogic < 0] = 0

    # done
    return pulseLogic

def _singlePulseFilter2(ppgRec):

    # low-pass filter
    [bb, aa] = butter(4, 5/ (ShimmerConstants.SAMPLING_RATE / 2), btype='lowpass')
    ppgRec = filtfilt(bb, aa, ppgRec)
    ppgRec = filtfilt(bb, aa, ppgRec)

    # step 2 - Condition signal
    printInfo("about to condition")
    #ppgRec_conditioned = conditionPPG(ppgRec)

    ppgSignal = ppgRec

    ppgSignal = normalizePPG(ppgSignal)
    ppgSignal[ppgSignal < 0.5] = 0
    ppgSignal = normalizePPG(ppgSignal)
    ppgSignal[ppgSignal < 0.5] = 0
    ppgSignal = normalizePPG(ppgSignal)

    ppgSignal[ppgSignal < 0.65] = 0
    ppgSignal[ppgSignal >= 0.65] = 1

    pulseLogic = np.diff(ppgSignal)
    # remove falling edge
    pulseLogic[pulseLogic < 0] = 0

    return pulseLogic

import matplotlib.pyplot as plt
def extractHRVandBPM_v2(ppgRec, recPath, experimental=False):
    tt = []
    HRV = []
    BPM = []
    peaks, properties = detect_peaks(ppgRec)

    tt = np.arange(0, ppgRec.shape[0]) / ShimmerConstants.SAMPLING_RATE
    beatTiming = tt[peaks[0::2]]
    tt = tt[peaks[0:-2:2]]
    ibi = np.diff(beatTiming)

    BPM = 1 / ibi * 60

    if(np.mean(BPM) < ShimmerConstants.MIN_BPM or np.mean(BPM) > ShimmerConstants.MAX_BPM):
        printError(f"[HeartRateUtils] HeartRate failed QA, mean BPM: {np.mean(BPM)} outside hardcoded limits [{ShimmerConstants.MIN_BPM},{ShimmerConstants.MAX_BPM}]")


        plt.figure(figsize=(10, 5))

        plt.plot(ppgRec, label='ppg conditioned', color='blue')
        plt.legend()

        ppg_features_dir = os.path.join(recPath, "features", "ppg")
        os.makedirs(ppg_features_dir, exist_ok=True)
        plt.savefig(os.path.join(ppg_features_dir, "ppg_rejected.png"))
        plt.show()
        plt.close()

        return None, None, None

    BPM[BPM < ShimmerConstants.MIN_BPM] = ShimmerConstants.MIN_BPM
    BPM[BPM > ShimmerConstants.MAX_BPM] = ShimmerConstants.MAX_BPM

    HRV = pd.DataFrame(data=ibi, columns=['HRV'])
    HRV = HRV.rolling(30).std() # might want to use normalized HRV instead

    return tt, HRV, BPM


# 2. Peak detection
def detect_peaks(ppg_signal, fs=51.2, distance=0.1, prominence=0.5):
    distance_samples = int(distance * fs)  # Convert distance to samples
    peaks, properties = signal.find_peaks(ppg_signal, distance=distance_samples, prominence=prominence)
    return peaks, properties

def extractHRVandBPM(ppgRec, experimental=False):

    # identify each pulse
    if experimental:
        singlePulseLogic = _singlePulseFilter2(ppgRec)
    else:
        singlePulseLogic = _singlePulseFilter(ppgRec)

    # generate timeline
    tt = np.arange(0, singlePulseLogic.shape[-1]) / ShimmerConstants.SAMPLING_RATE

    # compute basic HRV and BPM
    pulseTs = tt[singlePulseLogic == 1]
    
    if(len(pulseTs) < 2):
        return None, None, None
    
    HRV = np.diff(pulseTs)
    BPM = 1 / HRV * 60

    HRV = pd.DataFrame(data=HRV, columns=['HRV'])
    HRV = HRV.rolling(30).std() # might want to use normalized HRV instead

    # HRV might be bugged, need to validate

    ## smooth HRV
    #HRV = smooth(HRV, window_len=ShimmerConstants.HRV_WINDOW_WIDTH)

    # apply MIN-MAX heuristics to BPM
    BPM[BPM < ShimmerConstants.MIN_BPM] = ShimmerConstants.MIN_BPM
    BPM[BPM > ShimmerConstants.MAX_BPM] = ShimmerConstants.MAX_BPM

    # smooth BPM aggressively
    if BPM.shape[0]>ShimmerConstants.BPM_WINDOW_WIDTH:
        BPM = medfilt(smooth(smooth(BPM, window_len=ShimmerConstants.BPM_WINDOW_WIDTH), window_len=ShimmerConstants.BPM_WINDOW_WIDTH), ShimmerConstants.BPM_WINDOW_WIDTH)

    return pulseTs[:-1], HRV, BPM



def loadHR_HRV(filename):
    HR_HRVrec = []
    with(open(filename,"r")) as inputFile:
        reader = csv.reader(inputFile)
        for row in reader:
            HR_HRVrec.append([float(x) for x in row])

    return np.array(HR_HRVrec)


def loadBPM(src):
    data = []
    firstLine = True
    with open(src,'r') as hrFile:
        reader = csv.reader(hrFile)
        for row in reader:
            if firstLine:
                firstLine = False
                continue
            data.append([float(x) for x in row])

    return np.array(data)

def loadHR(basepath):
    data = []
    firstLine = True
    with open(basepath+"/results/BPM.csv",'r') as hrFile:
        reader = csv.reader(hrFile)
        for row in reader:
            if firstLine:
                firstLine = False
                header = row
                continue
            data.append([float(x) for x in row])

    return header, np.array(data)


