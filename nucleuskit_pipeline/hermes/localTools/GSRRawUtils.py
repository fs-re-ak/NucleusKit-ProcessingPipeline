
from nucleuskit_pipeline.hermes.HermesConstants import ShimmerConstants
from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError
import os,sys
import numpy as np
import pickle
#import hickle as hkl
from scipy import signal
import matplotlib.pyplot as plt
import csv
from nucleuskit_pipeline.shared.SignalProcessingUtils import smooth


# Warning text color and color reset
WARNING_COLOR = '\033[93m'
COLOR_RESET = '\033[0m'

# load GSR when presented as .tmp file, which is a csv encoding
def _loadGSRRaw(filename):
    rec = []
    with open(filename,"r") as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            rec.append([float(x) for x in row])
    return np.array(rec)

# load GSR from basepath
def loadGSR(basepath, showLoadedSignalInfo=False, showInvalidInfo=False):

    if os.path.isfile(basepath + "/wbData.pkl"):
        shimmerRec = np.array(pickle.load( open( basepath + "/wbData.pkl", "rb" ) ))

        # clean it from invalid samples
        to_remove = np.array([i for i, val in enumerate(shimmerRec) if len(val) != 6])
        if len(to_remove):
            shimmerRec = np.delete(shimmerRec, to_remove)

        GSR_INDEX = 4

    elif os.path.isfile(basepath + "/wbData.hkl"):
        shimmerRec = hkl.load(basepath + "/wbData.hkl")

        # clean it from invalid samples
        to_remove = np.array([i for i, val in enumerate(shimmerRec) if len(val) != 6])
        if len(to_remove):
            shimmerRec = np.delete(shimmerRec, to_remove)
        GSR_INDEX = 4

    elif os.path.isfile(basepath + "/gsr.tmp"):
        shimmerRec = _loadGSRRaw(basepath + "/gsr.tmp")
        GSR_INDEX = 1

    elif os.path.isfile(basepath + "/rawShimmer_0.csv"):
        shimmerRec = _loadGSRRaw(basepath + "/rawShimmer_0.csv")
        GSR_INDEX = 4

        if shimmerRec == []:
            printWarning(f"No wristband recording_data found, expected to find {basepath}/[wbData.pkl or /wbData.hkl or gsr.tmp or rawShimmer_0.csv")
            return None, None

    else:
        printWarning(f"No wristband recording_data found, expected to find {basepath}/[wbData.pkl or /wbData.hkl or gsr.tmp or rawShimmer_0.csv")
        return None, None

    if showInvalidInfo:
        # calculate invalid timestamps for later
        invalidTs = (to_remove-np.arange(len(to_remove)))/ShimmerConstants.SAMPLING_RATE + shimmerRec[0][0]
        printInfo('Invalid timestamps (for info) -----------------------')
        printInfo("Number of invalid: " + str(len(invalidTs)))
        printInfo("list of invalid: " + str(invalidTs))
        printInfo('-----------------------------------------------------')

    # convert to numpy array
    shimmerRec = np.array([np.array(x) for x in shimmerRec])

    # Check for recording_data validity
    if type(shimmerRec) is not np.ndarray:
        raise ValueError('GSR conversion to np.array failed, invalid values...')

    if not shimmerRec.size:
        printWarning("Wristband recording_data file is empty")
        return None, None

    # Show information about signal
    if showLoadedSignalInfo:
        printInfo('GSR Signal info')
        avgPeriod = float((np.average(shimmerRec[np.arange(10)+1,0] - shimmerRec[np.arange(10),0]))/1000.0)
        printInfo('Measured Sampling Rate is %.3f Hz' % (1.0/avgPeriod))
        printInfo('Defined Sampling Rate is %.3f Hz' % ShimmerConstants.SAMPLING_RATE)

        signalDuration = (shimmerRec.shape[0]/ShimmerConstants.SAMPLING_RATE)
        printInfo('Recording duration %.3f seconds, or %d minutes and %d seconds' % (signalDuration,int(signalDuration/60),int(signalDuration%60)))

    if(shimmerRec.shape[0]>0):
        gsrtime, gsrRec = shimmerRec[:,0]/1000, shimmerRec[:,GSR_INDEX]
    else:
        return None, None

    gsrtime -= gsrtime[0]
    gsrtime = np.arange(1,gsrtime.shape[0])/51.2
    
    return gsrtime, gsrRec



def conditionFullGSR(gsrRec, applyMedFilter=True):
    # low-pass filter
    [b, a] = signal.butter(ShimmerConstants.FILTER_ORDER, ShimmerConstants.GSR_BANDWIDTH[1]/ShimmerConstants.NYQUIST_FREQ, btype='lowpass')
    low_passed = signal.filtfilt(b, a, gsrRec)
    low_passed = signal.filtfilt(b, a, low_passed)

    if applyMedFilter:
        low_passed = signal.medfilt(low_passed, 15)
    return low_passed

def conditionGSRHard(gsrRec, applyMedFilter=True):
    # low-pass filter
    [b, a] = signal.butter(ShimmerConstants.FILTER_ORDER, ShimmerConstants.GSR_BANDWIDTH[1]/ShimmerConstants.NYQUIST_FREQ, btype='lowpass')
    low_passed = signal.filtfilt(b, a, gsrRec)
    low_passed = signal.filtfilt(b, a, low_passed)

    low_passed = signal.medfilt(low_passed, 101)
    low_passed = signal.medfilt(low_passed, 101)
    low_passed = signal.medfilt(low_passed, 101)
    low_passed = smooth(low_passed,window_len=31)

    if applyMedFilter:
        low_passed = signal.medfilt(low_passed, 15)
    return low_passed

def conditionTonicGSR(gsrRec, applyMedFilter=True):
    # low-pass filter
    [b, a] = signal.butter(ShimmerConstants.FILTER_ORDER, ShimmerConstants.TONIC_FC/ShimmerConstants.NYQUIST_FREQ, btype='lowpass')
    low_passed = signal.filtfilt(b, a, gsrRec)
    low_passed = signal.filtfilt(b, a, low_passed)

    if applyMedFilter:
        low_passed = signal.medfilt(low_passed, 15)
    return low_passed

def extractGSRFeatures(gsrRec, tightThreshold=False):

    peaks = []
    types = []

    for i in range(gsrRec.shape[0]):
        if i<4 or i>(gsrRec.shape[0]-5):
            continue

        if np.mean([gsrRec[i-4],gsrRec[i-3],gsrRec[i-2],gsrRec[i-1]]) < gsrRec[i] and gsrRec[i] >= np.mean([gsrRec[i+4],gsrRec[i+3],gsrRec[i+2],gsrRec[i+1]]):
            peaks.append(i)
            types.append("max")

        if np.mean([gsrRec[i-4],gsrRec[i-3],gsrRec[i-2],gsrRec[i-1]]) > gsrRec[i] and gsrRec[i] <= np.mean([gsrRec[i+4],gsrRec[i+3],gsrRec[i+2],gsrRec[i+1]]):
            peaks.append(i)
            types.append("min")

    types = np.array(types)
    peaks = np.array(peaks)

    lineYs = [np.min(gsrRec), np.max(gsrRec)]

    for i in range(0,peaks.shape[-1]-1):
        if types[i] == types[i+1] and types[i+1] == "max":
            peaks[i] = 0
            types[i] = "null"

        if types[i] == types[i+1] and types[i+1] == "min":
            peaks[i] = 0
            types[i] = "null"


    types = types[peaks!=0]
    peaks = peaks[peaks!=0]

    if types[0] == "min":
        peaks = peaks[1:]
        types = types[1:]

    if types[-1] == "max":
        peaks = peaks[:-1]
        types = types[:-1]


    for i in range(0, peaks.shape[-1]-1, 2):

        if tightThreshold:
            if np.abs(gsrRec[peaks[i]]-gsrRec[peaks[i+1]])<ShimmerConstants.GSR_THRESHOLD_TIGHT:
                peaks[i] = 0
                types[i] = "null"
                peaks[i+1] = 0
                types[i+1] = "null"

        else:
            if gsrRec[peaks[i]] < 200:
                if np.abs(gsrRec[peaks[i]] - gsrRec[peaks[i + 1]]) < 1:
                    peaks[i] = 0
                    types[i] = "null"
                    peaks[i+1] = 0
                    types[i+1] = "null"

            elif gsrRec[peaks[i]] < 400:
                if np.abs(gsrRec[peaks[i]] - gsrRec[peaks[i + 1]]) < 2:
                    peaks[i] = 0
                    types[i] = "null"
                    peaks[i+1] = 0
                    types[i+1] = "null"

            else:
                if np.abs(gsrRec[peaks[i]] - gsrRec[peaks[i + 1]]) < 3:
                    peaks[i] = 0
                    types[i] = "null"
                    peaks[i+1] = 0
                    types[i+1] = "null"

            #else:
            #    print(repr(gsrRec[peaks[i]]) + ":" + repr(gsrRec[peaks[i+1]]))

    types = types[peaks!=0]
    peaks = peaks[peaks!=0]

    # amplitude is max - min
    amplitudes = gsrRec[peaks[::2]] - gsrRec[peaks[1::2]]
    # amplitude is time of max
    startingPoints = gsrRec[peaks[::2]]
    # slopetime is time between min-max (done as such to give positive values)
    slopeTimes = np.array(peaks[1::2]-peaks[0::2]).astype(float)
    # IEI is time between the min(t) and max (t+1), the very first is set to 0, because there are
    # no preceding event
    IEI = np.r_[np.nan, np.array(peaks[2::2]-peaks[1:-1:2]).astype(float)]

    # conversion to milliseconds time base
    slopeTimes /= ShimmerConstants.SAMPLING_RATE
    slopeTimes *= 1000
    IEI /= ShimmerConstants.SAMPLING_RATE
    IEI *= 1000

    # quantilization of gsrEvents based on amplitude
    amplitudesQT = gsrRec[peaks[::2]] - gsrRec[peaks[1::2]]
    quantiles = np.percentile(amplitudesQT,[0, 25, 50, 75, 100])

    norm = [0.25, 0.5, 0.75, 1.0]
    for i in range(quantiles.shape[-1]-1):
        amplitudesQT[np.logical_and(amplitudesQT>=quantiles[i],amplitudesQT<=quantiles[i+1])] = norm[i]
    peaks = peaks[::2]

    return peaks, amplitudesQT, amplitudes, startingPoints, slopeTimes, IEI


def writeGSREventsFeatures(peaks, amplitudes, startingPoints, slopeTimes, IEI, dest):
    with open(dest,'w',newline='') as arousalFile:
        writer = csv.writer(arousalFile)
        for i in range(peaks.shape[-1]):
            writer.writerow([peaks[i], amplitudes[i], startingPoints[i], slopeTimes[i], IEI[i]])


def loadGSREventsFeatures(src):
    data = []
    firstLine = True
    with open(src,'r') as gsrEventsFile:
        reader = csv.reader(gsrEventsFile)
        for row in reader:
            if firstLine:
                firstLine = False
                continue
            data.append([float(x) for x in row])

    return np.array(data)


def writeTonicEDA(gsrRec, gsrTonicRec, dest):
    with open(dest,'w',newline='') as arousalFile:
        writer = csv.writer(arousalFile)
        for i in range(gsrRec.shape[-1]):
            writer.writerow([gsrRec[i], gsrTonicRec[i]])


# ============================================================
# Arousal Results I/O Functions
# (Merged from ArousalUtils.py for consolidation)
# ============================================================

def writeArousalResults(arousal, dest):
    """
    Write arousal results to CSV file.
    
    Args:
        arousal: Arousal data array
        dest: Destination file path
    """
    with open(dest, 'w', newline='') as arousalFile:
        writer = csv.writer(arousalFile)
        for sample in arousal:
            writer.writerow(sample)


def loadArousal(basepath):
    """
    Load arousal results from CSV file.
    
    Args:
        basepath: Base path to recording directory
        
    Returns:
        np.ndarray: Arousal data
    """
    data = []
    firstLine = True
    with open(basepath + "/results/arousal.csv", 'r') as arousalFile:
        reader = csv.reader(arousalFile)
        for row in reader:
            if firstLine:
                firstLine = False
                continue
            data.append([float(x) for x in row])
    
    return np.array(data)


def writeGSREventsResults(peaks, amplitudes, dest):
    """
    Write GSR event peaks and amplitudes to CSV file.
    
    Args:
        peaks: Peak timestamps
        amplitudes: Peak amplitudes  
        dest: Destination file path
    """
    with open(dest, 'w', newline='') as arousalFile:
        writer = csv.writer(arousalFile)
        for i in range(peaks.shape[-1]):
            writer.writerow([peaks[i], amplitudes[i]])


def loadGSREvents(filename):
    """
    Load GSR events from CSV file.
    
    Args:
        filename: Path to GSR events file
        
    Returns:
        np.ndarray: GSR events data, or None if file not found
    """
    try:
        with open(filename, 'r') as gsrEventsFile:
            gsrEvents = []
            rows = csv.reader(gsrEventsFile)
            for row in rows:
                gsrEvents.append([float(x) for x in row])
        return np.array(gsrEvents)
    except:
        return None

