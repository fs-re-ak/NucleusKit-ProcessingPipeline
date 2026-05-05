import collections
import csv
import os
import shutil
from datetime import datetime
from os import listdir, path, walk

import numpy as np
import pandas as pd

from nucleuskit_pipeline.logging_utils import printInfo, printWarning


def loadEvents(fullFilename):
    events = pd.read_csv(fullFilename)
    a = events.loc[events["EventName"]=="OFFSET"]

    if len(a) > 0:
        events["Start"] -= int(a["Start"])
        events["End"] -= int(a["Start"])

    if "oasis" in fullFilename.lower():
        # Convert miliseconds to seconds
        events["Start"] /= 1000
        events["End"] /= 1000

    try:
        events["Value"] = events["Value"].apply(lambda x: x.replace(']', "").replace('[', ""))
    except Exception:
        # This is a fix for the event file create by Sayeed for OASIS
        events["Value"] = events["Value"].replace(']', "").replace('[', "")

    #events["Start"].to_numeric()
    #events["End"].to_numeric()

    events = events.loc[np.logical_and(events["EventName"] != "OFFSET", events["EventName"] != "FIN")]

    return events

def class_iter(Class):
    return (elem for elem in dir(Class) if elem[:2] != '__')


def getFilepath(cachePath, suffix):

    lastdir = os.path.dirname(cachePath).split('/')[-1]

    if lastdir == 'rawData':
        datapath = cachePath + 'rawData/' + suffix
    else:
        rawDatapath = cachePath + 'rawData/'

        if os.path.isdir(rawDatapath):
            datapath = cachePath + 'rawData/' + suffix

        else:
            datapath = cachePath + suffix

    return datapath


def get_recording_directories(dirname):
    subfolders = list()

    for root, dirs, files in walk(dirname):
        # todo, qualification happens before preparators of directories
        if (not dirs) or (dirs==['experimentConfigs', 'features', 'rawData', 'results']) or ('rawData' in dirs):
            subfolders.append(root.replace("\\","/"))

    subfolders = [recpath for recpath in subfolders if not
                    (recpath.endswith('rawData') or recpath.endswith('features') or
                    recpath.endswith('results') or recpath.endswith('experimentConfigs'))]

    # Make sure the correct file separator is used in path strings
    subfolders = [recpath.replace("\\", path.sep).replace("/", path.sep) for recpath in subfolders]

    return subfolders


def createDirectories(ACTIVE_DATASET, configs):
    try:
        BASE_PATH = configs.DATA_PATH
        OUT_PATH = configs.OUT_PATH
    except Exception:
        BASE_PATH = configs['BASE_PATH']
        OUT_PATH = configs['OUT_PATH']
    createFolderStructure(configs)
    resultsPath = OUT_PATH + ACTIVE_DATASET + '/'
    createFolderStructure(configs, resultsPath)

    FIG_PATH = resultsPath + 'figures/'

    figPath = FIG_PATH
    createFolderStructure(configs, figPath)

    datasetPath = BASE_PATH + ACTIVE_DATASET + '/'

    return datasetPath



def selectDataset(ACTIVE_DATASET, configs):

    if ACTIVE_DATASET is not None:
        return ACTIVE_DATASET

    else:
        try:
            datasets = [d for d in listdir(configs.DATA_PATH)]
        except Exception:
            datasets = [d for d in listdir(configs['BASE_PATH'])]
        printInfo("--------------------------------------------------------------------------------")
        printInfo("Current available datasets ...")

        for i, entry in enumerate(datasets):
            printInfo(f"  Dataset {i+1}: {entry}")

        printInfo("  Dataset 0: All the above")
        printInfo("Select a dataset using the dataset number ...")
        printInfo("--------------------------------------------------------------------------------")

        ans = int(input())
        dataset = [datasets[ans-1]] if ans else datasets

        #print("\n")
        printInfo(f"Dataset: {dataset}")

        return dataset


_SESSION_ROOT_RESERVED_NAMES = frozenset(
    {"rawData", "features", "results", "experimentConfigs", "meta"}
)


def ensure_session_rawdata_layout(session_root: str) -> None:
    """
    If ``rawData`` is missing under ``session_root``, create it and move every other
    top-level entry into it (excluding standard pipeline folders).
    """
    session_root = os.path.abspath(session_root)
    raw_data_path = path.join(session_root, "rawData")
    if os.path.isdir(raw_data_path):
        return
    if os.path.exists(raw_data_path):
        return
    file_contents = os.listdir(session_root)
    os.makedirs(raw_data_path, exist_ok=True)
    for name in file_contents:
        if name in _SESSION_ROOT_RESERVED_NAMES:
            continue
        shutil.move(path.join(session_root, name), raw_data_path)


def prepareDirectory(recpath):
    printInfo("[fileTools] Preparing directories")
    ensure_session_rawdata_layout(recpath)

    if not os.path.exists(recpath + "/features"):
        os.makedirs(recpath + "/features")
    if not os.path.exists(recpath + "/results"):
        os.makedirs(recpath + "/results")
    if not os.path.exists(recpath + "/meta"):
        os.makedirs(recpath + "/meta")
    if not os.path.exists(recpath + "/experimentConfigs"):
        os.makedirs(recpath + "/experimentConfigs")

    curr_time = datetime.now().strftime("%d_%m_%y-%H_%M_%S")

    try:
        processingLog = open(recpath + '/results/processingLog.txt', 'w')
        processingLog.write('Processing log for Analysis started at ' + curr_time)
        processingLog.close()
    except OSError:
        printWarning("[fileTools.py::prepareDirectory] Cannot write to processingLog.txt, skipping")


def printToFile(recpath, line):

    try:
        processingLog = open(recpath + '/results/processingLog.txt', 'a')
        processingLog.write(line+"\n")
        processingLog.close()
    except OSError:
        printWarning("[fileTools.py::printToFile] Cannot write to processingLog.txt, skipping")


def createFolderStructure(configs, recpath=False):
    try:
        BASE_PATH = configs.DATA_PATH
        OUT_PATH = configs.OUT_PATH
    except Exception:
        BASE_PATH = configs['BASE_PATH']
        OUT_PATH = configs['OUT_PATH']
    if recpath:
        if not os.path.isdir(recpath):
            os.mkdir(recpath)
    else:
        if not os.path.isdir(BASE_PATH):
            os.mkdir(BASE_PATH)
        if not os.path.isdir(OUT_PATH):
            os.mkdir(OUT_PATH)



def loadCSVAsNumpy(filename, hasHeader=False):
    """
    Wrapper over csv to load a float-only csv file into a
    numpy array.

    :param filename: path and filename
    :param hasHeader: True is need to skip header line
    :return: numpy array containing values
    """

    data = []

    with open(filename, "r") as csvFile:
        reader = csv.reader(csvFile)

        for row in reader:

            if hasHeader:
                hasHeader = False
            else:
                data.append([float(x) for x in row])


    return np.array(data)


def buildEmotionsDataframe(recpath, configs):

    # don't build the dataset if it hasn't been processed
    emotions_csv = path.join(recpath, "results", "Emotions.csv")
    if not os.path.isfile(emotions_csv):
        return

    # don't build the dataset if the completed tag is found
    # if os.path.isfile(cacherecpath + "/results/" + pipeName + "/results_tag.COMPLETED"):
    #    return

    # make an empty dataframe which we will append to, if it's not there yet
    if not os.path.isfile(recpath + "emotionsDataframe.pickle"):
        _cols = [
            "Dataset_Name",
            "Dataset_Number",
            "Algorithm",
            "DataType",
            "TimeStamp",
            "Emotion",
            "Probability",
        ]
        df = pd.DataFrame(columns=_cols)
        df.to_pickle(recpath + "emotionsDataframe.pickle")

    current_dataset = pd.read_csv(emotions_csv)
    current_emotions = current_dataset.columns[1:]
    number_emotions = current_emotions.__len__()
    number_of_samples = current_dataset['Timestamp'].count()

    df = pd.read_pickle(recpath + "emotionsDataframe.pickle")
    name_col = "Dataset_Name" if "Dataset_Name" in df.columns else "Dataset Name"
    _ = df[name_col].count()
    _ = number_emotions * number_of_samples * 2

    folder_lvl = recpath.split('/').__len__()-1

    datasetName = recpath.split('/')[folder_lvl]
    datasetNumber = recpath.split('/')[folder_lvl + 1] + '/' + recpath.split('/')[folder_lvl + 2]

    tempDict = collections.OrderedDict()
    i=0
    for sample in range(0, number_of_samples):
        for emotion in current_emotions:
            emotionProb = current_dataset[emotion].iloc[sample]
            label = 0.0
            if emotionProb == current_dataset.iloc[sample][1:].max():
                label = 1.0
            tempDict[i] = {'Dataset_Name': datasetName, 'Dataset_Number': datasetNumber,
                           'DataType': 'Prob', 'TimeStamp': sample, 'Emotion': emotion,'Probability': emotionProb}
            tempDict[i+1] = {'Dataset_Name': datasetName, 'Dataset_Number': datasetNumber,
                             'DataType': 'Label', 'TimeStamp': sample, 'Emotion': emotion,'Probability': label}
            i = i + 2

    df = pd.concat([df,pd.DataFrame.from_dict(tempDict, "index")])
    df.to_pickle(recpath + "emotionsDataframe.pickle")


def buildDataframe(recpath, configs):
    emoData = None
    engageData = None
    asymmetryData = None
    arousalData = None
    bpmhrvData = None
    events = None

    ds = configs.DATA_PATH.split('/')[-1]
    #recording = recpath.EXP_PATH.split('/')[-1]
    recording = recpath.split('/')[-1]

    debugHdr = f"{configs.DF_ASSEMBLY_HDR}[{ds}] ({recording}): "

    resultsDirectory = recpath

    # load the feature dataframes
    emotions_path = path.sep.join([resultsDirectory, "Emotions.csv"])
    if os.path.isfile(emotions_path):
        emoData = pd.read_csv(emotions_path)

    else:
        printInfo(f"{debugHdr}No file 'Emotions.csv'")

    engage_path = path.sep.join([resultsDirectory, "Engagement.csv"])
    if os.path.isfile(engage_path):
        engageData = pd.read_csv(engage_path)

    else:
        printInfo(f"{debugHdr}No file 'Engagement.csv'")

    asymmetry_path = path.sep.join([resultsDirectory, "frontalAsymmetry.csv"])
    if os.path.isfile(asymmetry_path):
        asymmetryData = pd.read_csv(asymmetry_path)

    else:
        printInfo(f"{debugHdr}No file 'frontalAsymmetry.csv'")

    arousal_path = path.sep.join([resultsDirectory, "Arousal.csv"])
    arousal_path2 = path.sep.join([resultsDirectory, "arousal.csv"])
    if os.path.isfile(arousal_path):
        arousalData = pd.read_csv(arousal_path)
    elif os.path.isfile(arousal_path2):
        arousalData = pd.read_csv(arousal_path2)
    else:
        printInfo(f"{debugHdr}No file 'Arousal.csv' or 'arousal.csv'")

    bpmhrv_path = path.sep.join([resultsDirectory, "BPM_HRV.csv"])
    if os.path.isfile(bpmhrv_path):
        bpmhrvData = pd.read_csv(bpmhrv_path)

    else:
        printInfo(f"{debugHdr}No file 'BPM_HRV.csv'")

    def _nrows(d):
        if d is None:
            return 0
        return int(d.shape[0])

    feature_length = int(
        np.max(
            [
                _nrows(emoData),
                _nrows(engageData),
                _nrows(asymmetryData),
                _nrows(arousalData),
                _nrows(bpmhrvData),
            ]
        )
    )
    if feature_length == 0:
        printWarning(f"{debugHdr}No feature files found; skipping Dataframe assembly.")
        return

    # Merge datasets that could be read
    allValues = pd.DataFrame()
    allValues["Recording"] = [recording] * feature_length
    allValues["Session"] = [ds] * feature_length

    try:
        allValues = pd.concat([engageData["Timestamp"], allValues], axis=1)
    except Exception:
        pass
    try:
        allValues = pd.concat([emoData[configs.EMOTIONS], allValues], axis=1)
    except Exception:
        pass
    try:
        allValues = pd.concat([engageData["Engagement"], allValues], axis=1)
    except Exception:
        pass
    try:
        allValues = pd.concat([asymmetryData["FrontalAsymmetry"], allValues], axis=1)
    except Exception:
        pass
    try:
        allValues = pd.concat([arousalData["arousal"], allValues], axis=1)
    except Exception:
        pass
    try:
        allValues = pd.concat([bpmhrvData[['HRV', 'BPM']], allValues], axis=1)
    except Exception:
        pass

    # Apply manually marked events
    events_path = path.sep.join(
        [configs.DATA_PATH + ds, recording, "features", "processedFeatureEvents.csv"]
    )
    try:
        events = loadEvents(events_path)
    except Exception:
        printInfo(f"{debugHdr}No file {events_path}")

    allValues["Tag"] = ["NULL"] * feature_length
    if events is not None:
        for _, row in events.iterrows():
            mask = allValues["Timestamp"].between(row["Start"], row["End"])
            allValues.loc[mask, "Tag"] = row["EventName"]
            tag_id = row["Value"] if row["Value"] != "" else "NULL"
            allValues.loc[mask, "TagID"] = tag_id

    allValues = allValues[allValues.Tag != "NULL"]
    allValues = allValues[allValues.Tag != "INVALID"]

    saveLocation = path.sep.join([configs.OUT_PATH, 'Dataframe.csv'])
    allValues["Timestamp"] = np.round(allValues["Timestamp"], 3)

    allValues.to_csv(saveLocation, na_rep='NULL', index=False)
