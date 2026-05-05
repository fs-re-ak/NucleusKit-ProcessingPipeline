
import os,sys
import numpy as np
import json
from nucleuskit_pipeline.session.layout import loadCSVAsNumpy
from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError

class RecordingInfo():
    """
    This is implemented as a class, in case we need to carry it later
    """

    basepath = None
    duration = None
    eegReadable = False
    shimmerReadable = False
    povCameraPresent = False
    uwbValid = False
    gpsValid = False

    def __init__(self, basepath, load=False):

        self.basepath = basepath

        if not load:
            # get length
            self._getRecordingLengthInfo()

            # save to file
            printInfo(json.dumps(self._save(), indent=4))

        else:
            self._load()



    def _runSanityCheck(self):
        pass

    def _detectFilePresence(self):
        pass

    def _getRecordingLengthInfo(self):

        # check the length of the allrecordings (in time)
        try:
            if os.path.exists(os.path.join(self.basepath, "rawData/rawShimmer_0.csv")):
                shimmerRecording = loadCSVAsNumpy(os.path.join(self.basepath, "rawData/rawShimmer_0.csv"))
            else:
                shimmerRecording = loadCSVAsNumpy(os.path.join(self.basepath, "rawData/gsr.tmp"))
            shimmerDuration = shimmerRecording[-1, 0] - shimmerRecording[0, 0]
            self.shimmerReadable = True
        except:
            shimmerDuration = 0

        try:
            if os.path.exists(os.path.join(self.basepath, "rawData/rawEEG_0.csv")):
                eegRecording = loadCSVAsNumpy(os.path.join(self.basepath, "rawData/rawEEG_0.csv"))
            else:
                eegRecording = loadCSVAsNumpy(os.path.join(self.basepath, "rawData/eeg.tmp"))
            eegDuration = eegRecording[-1, 0] - eegRecording[0, 0]
            self.eegReadable = True
        except:
            eegDuration = 0

        # use shimmer and EEG as reference, select the longest one
        self.duration = np.max([shimmerDuration, eegDuration])/1000
        self.duration = self._myround(self.duration, base=0.5)

        if self.duration == 0:
            printWarning("Warning, duration is equal to 0... need at least the shimmer or eeg to provide ground truth")

        pass

    def _myround(self, x, base=5):
        return base * round(x / base)

    def _save(self):

        data = {}
        data['basepath'] = str(self.basepath)
        data['duration'] = str(self.duration)
        data['eegReadable'] = str(self.eegReadable)
        data['shimmerReadable'] = str(self.shimmerReadable)
        data['povCameraPresent'] = str(self.povCameraPresent)
        data['uwbValid'] = str(self.uwbValid)
        data['gpsValid'] = str(self.gpsValid)

        with open(os.path.join(self.basepath, "features/metainfo.json"), "w") as outfile:
            json.dump(data, outfile, indent=4)
        return data

    def _load(self):
        with open(os.path.join(self.basepath, "features/metainfo.json"), "r") as infile:
            data = json.load(infile)

            self.basepath = data['basepath']
            self.duration = float(data['duration'])
            self.eegReadable = bool(data['eegReadable'])
            self.shimmerReadable = bool(data['shimmerReadable'])
            self.povCameraPresent = bool(data['povCameraPresent'])
            self.uwbValid = bool(data['uwbValid'])
            self.gpsValid = bool(data['gpsValid'])


    def generateRecordingInfo(self):
        pass

    def _toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)



def extractMetaInfo(basepath):

    printInfo("[metaInfoTools] Extracting metainfo ...]")

    recInfo = RecordingInfo(basepath)
    #recInfo = RecordingInfo(basepath, load=True)


