from nucleuskit_pipeline.hermes.HermesConstants import ShimmerConstants
from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError
import os
#import hickle as hkl
import numpy as np
from scipy import signal
import csv
import pandas as pd


def loadPPGRaw(filename):
    rec = []
    with open(filename,"r") as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            rec.append([float(x) for x in row])

    return np.array(rec)



def loadPPG(basepath, showLoadedSignalInfo=False, showInvalidInfo=False):

    if os.path.isfile(basepath + "/ppg.tmp"):
        shimmerRec = loadPPGRaw(basepath + "/ppg.tmp")
        PPG_INDEX = 1

    elif os.path.isfile(basepath + "/rawShimmer_0.csv"):
        shimmerRec = loadPPGRaw(basepath + "/rawShimmer_0.csv")
        PPG_INDEX = 5

    elif os.path.isfile(basepath + "/shimmer.csv"):
        shimmerRec = loadPPGRaw(basepath + "/shimmer.csv")
        PPG_INDEX = 5
    else:
        return None, None

    # convert to numpy array
    shimmerRec = np.array([np.array(x) for x in shimmerRec])

    # Check for recording_data validity
    if type(shimmerRec) is not np.ndarray:
        raise ValueError('GSR conversion to np.array failed, invalid values...')

    #print(shimmerRec.shape)
    if shimmerRec.shape[0]==0:
        return None, None
    ppgtime, ppgRec = shimmerRec[:, 0], shimmerRec[:, PPG_INDEX]
    ppgtime -= ppgtime[0]
    ppgtime /= 1000

    return ppgtime, ppgRec


def conditionPPG(ppgRec):

    low = ShimmerConstants.GSR_BANDWIDTH[0] / ShimmerConstants.NYQUIST_FREQ
    high = 10 / ShimmerConstants.NYQUIST_FREQ
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, ppgRec)
    return filtered_signal

    return band_passed



def writePPGEventsFeatures(tt, BPM, HRV, path):

    # for BPM
    BPMrec = np.c_[tt, BPM]

    # convert to dataframe
    df_BPM = pd.DataFrame(data=BPMrec, columns=["Time", "BPM"])
    df_BPM.Time -= df_BPM.Time[0]
    df_BPM = df_BPM.set_index(pd.to_datetime(df_BPM['Time'], unit='s'), drop=False)

    # resample
    resample_index = pd.date_range(start=df_BPM.index[0], end=df_BPM.index[-1], freq='500ms')
    dummy_frame = pd.DataFrame(np.nan, index=resample_index, columns=df_BPM.columns)
    df_BPM = df_BPM.combine_first(dummy_frame).interpolate('time').iloc[:]
    df_BPM = df_BPM[(df_BPM.Time * 2 % 1) == 0.0]

    #for HRV
    HRVrec = np.c_[tt, HRV]

    # convert to dataframe
    df_HRV = pd.DataFrame(data=HRVrec, columns=["Time", "HRV"])
    df_HRV.Time -= df_HRV.Time[0]
    df_HRV = df_HRV.set_index(pd.to_datetime(df_HRV['Time'], unit='s'), drop=False)

    # resample
    resample_index = pd.date_range(start=df_HRV.index[0], end=df_HRV.index[-1], freq='500ms')
    dummy_frame = pd.DataFrame(np.nan, index=resample_index, columns=df_HRV.columns)
    df_HRV = df_HRV.combine_first(dummy_frame).interpolate('time').iloc[:]
    df_HRV = df_HRV[(df_HRV.Time * 2 % 1) == 0.0]

    df_BPM['BPM'] = df_BPM['BPM'].rolling(window=15, center=True).mean()
    df_BPM['BPM'] = df_BPM['BPM'].rolling(window=15, center=True).mean()
    df_BPM['BPM'] = df_BPM['BPM'].rolling(window=15, center=True).mean()
    df_BPM['BPM'] = df_BPM['BPM'].rolling(window=15, center=True).mean()

    df_HRV['HRV'] = df_HRV['HRV'].rolling(window=15, center=True).mean()
    df_HRV['HRV'] = df_HRV['HRV'].rolling(window=15, center=True).mean()

    numpy_version = np.c_[df_BPM.Time, df_BPM.BPM, df_HRV.HRV]

    #import matplotlib.pyplot as plt
    #plt.plot(numpy_version[:, 0], numpy_version[:, 1])
    #plt.show()

    with open(path + "/results/BPM_HRV.csv", "w", newline="") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["Timestamp", "BPM", "HRV"])

        for i in range(numpy_version.shape[0]):
            writer.writerow(numpy_version[i])

