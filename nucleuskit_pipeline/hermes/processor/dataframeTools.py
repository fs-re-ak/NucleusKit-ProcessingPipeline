from nucleuskit_pipeline.hermes.HermesConstants import HermesConstants
from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError
import numpy as np
import datetime
import pandas as pd

def loadRecordingStarttime(cachepath):
    pcRefTS, rockRefTS = datetime.datetime.now(), datetime.datetime.utcnow()

    info = np.loadtxt(cachepath + "info.txt",'str')
    recyear = int(info[1].split('-')[0])
    recmonth = int(info[1].split('-')[1])
    recday = int(info[1].split('-')[2])
    recstart = info[1] + '-' + info[2]
    recstartdt = datetime.datetime.strptime(recstart,'%Y-%m-%d-%H:%M:%S.%f')

    recstartdt += (pcRefTS - rockRefTS)

    virtstartdt = datetime.datetime(year=recyear, month=recmonth, day=recday, hour=0, minute=0, second=0)

    return recstartdt, virtstartdt



def appendToDataframe(mean_PowerBands, percent_PowerBands, windows):
    channel_names = HermesConstants.CHANNEL_NAMES
    eegchans = HermesConstants.EEG_CHANNELS
    nchans = len(eegchans)

    tmp = {}
    counter=0

    printInfo("Appending power to dataframe ... ")
    for i, win_start in enumerate(windows):

        for j in range(nchans):
            curr_chan = channel_names[eegchans[j]]

            for k, band in enumerate(HermesConstants.POWER_BANDS):

                tmp[counter] = {"Time_Seconds": win_start, "Channel": curr_chan, "Power_Band": band, "Mean_Power": mean_PowerBands[i,j,k], "Percent_Power": percent_PowerBands[i,j,k]}

                counter += 1

    df = pd.DataFrame.from_dict(tmp, "index")

    # append recording dataset to full dataset
    return df



















