"""
WristbandManager
    - Top-level EEG Wristband manager 

Roles:
    - Provide a simplified interface for EEG headset
    - Implements a level of abstraction, over all supported headsets:
        - Fake Muse
        - Muse
    
Design:
    - is observable

TODO: 
    - setup recorder is a blocking call, this hangs the other recorders setup 
    
Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Spring 2018
"""

from time import sleep
from threading import Thread, Semaphore, Lock
from recorderInterface.wristbandInterface.WristbandAdapter import WristbandAdapterFactory
from recorderInterface.wristbandInterface.WristbandProxy import WristbandProxyFactory
from systemController.SystemStatus import SystemStatus, SYSTEM_STATUS
from recorderInterface.AbstractRecorder import AbstractRecorder

from outStreamInterface.StreamsController import OutStreamsController

class WristbandManager(AbstractRecorder):
    
    proxy = None
    adapter = None
        
    observerCallback = []
    
    t = None
    startThread = None
    
    recorderSem = None
    startRecordingLock = None
    
    paired = False

    museHandle = None
    
    """
        Name: __init__
        Descr: constructor
        Parameters:
            configuration: string indicative of desired configuration
    """
    def __init__(self, configuration):
        super().__init__()
        
        # instantiate system, based on requested configuration
        if configuration == 'SHIMMER-IO':
            self.proxy = WristbandProxyFactory.factory('SHIMMER-IO', self);
            self.adapter = WristbandAdapterFactory.factory('SHIMMER-IO', self);
            print('[WristbandManager.py] Shimmer IO interface selected')
        elif configuration == 'SHIMMER-SERIAL':
            self.proxy = WristbandProxyFactory.factory('SHIMMER-SERIAL', self);
            self.adapter = WristbandAdapterFactory.factory('SHIMMER-SERIAL', self);
            print('[WristbandManager.py] Shimmer direct serial interface selected')
        else:
            print('[WristbandManager.py] invalid factory configuration')
            
        self.recorderSem = Semaphore(value=0)
        self.startRecordingLock = Lock()
        
        # link adapter <> proxy together
        self.proxy.attachAdapter(self.adapter)
        self.adapter.attachProxy(self.proxy)

    def attachOutRawDataStream(self, outRawDataStreamHandle):
        self.adapter.attachOutRawDataStream(outRawDataStreamHandle)

    # will start pairing task, there can only be one
    def setupRecorder(self):

        outStreamCtl = OutStreamsController()
        self.attachOutRawDataStream(outStreamCtl.getStream("OUT_RAW_SHIMMER_FILE"))

        # SystemStatus.setStatus(SYSTEM_STATUS.PAIRING, True)
        while True:
            try:
                self.adapter.connect()  # blocking call
                break
            except Exception as e:
                print(f"[WristbandManager.py] Error connecting to wristband: {e}")
                sleep(2)
        # SystemStatus.setStatus(SYSTEM_STATUS.PAIRING, False, broadcast=False)
        # SystemStatus.setStatus(SYSTEM_STATUS.PAIRED, True)
        self.paired = True
            
        if self.startThread is None:
            self.startThread = Thread(target=self.startupTask,)
            self.startThread.daemon = True
            self.startThread.start()
        
        if self.watchdogThread is None:
            self.watchdogThread = Thread(target=self._watchdogTask,)
            self.watchdogThread.daemon = True
            self.watchdogThread.start()
    
    # call to start recording, needs session path and startDelay
    def startRecording(self, sessionPath, startDelay=1):
        self.startRecordingLock.acquire()
        
        # can start, if not recording already
        if self.recording == False and self.paired == True:
            print('[WristbandManager.py] starting recording')
            self.startDelay = startDelay    
            self.recording = True
            self.sessionPath = sessionPath
            self.recorderSem.release()
        else:
            print('[WristbandManager.py] attempt at starting second screen recorder')
            self.startRecordingLock.release()
        
        # update observers on status changed
        self.statusChanged()
        
    # call to stop recording
    def stopRecording(self):
        self.startRecordingLock.acquire()
        
        # can only stop if already recording
        if self.recording == True:
            self.adapter.stopRecording() 
            self.recording = False
        else:
            print('[WristbandManager.py] attempt at stoping a non recording recorder')
        
        self.startRecordingLock.release()
        self.statusChanged()
        pass

    def attachMuseHandle(self, museHandle):
        self.museHandle = museHandle

    
    def wristbandDisconnected(self):
        print('[WristbandManager.py] attempt at re-pairing with device')
        self.paired = False
        # SystemStatus.setStatus(SYSTEM_STATUS.PAIRING, True, broadcast=False)
        # SystemStatus.setStatus(SYSTEM_STATUS.PAIRED, False)

        #if self.museHandle is not None:
        #    while not self.museHandle.isPaired:
        #        sleep(1)

        self.adapter.connect()
        print('[WristbandManager.py] re-pairing with device succeeded')
        
        self.paired = True
        # SystemStatus.setStatus(SYSTEM_STATUS.PAIRING, False, broadcast=False)
        # SystemStatus.setStatus(SYSTEM_STATUS.PAIRED, True)

        pass

    # initialization priority
    def getPriority(self):
        return WristbandManager.PRIORITY_HIGH
    
    # nothing to do
    def postProcess(self):
        # nothing to do
        self.adapter.postProcess()
        pass

       
    # when semaphoe is acquired, 
    #   - starts the recorder
    #     a delay to startup might have been configured,
    #     if this is the case, this thread will handle the delay such
    #     that the startRecording is non-blocking
    def startupTask(self):
        while True:
            self.recorderSem.acquire()
            print('[WristbandManager.py] startRecording')
            self.adapter.setupRecorder(self.sessionPath)
            self.adapter.startRecording(self.startDelay)
            self.startRecordingLock.release()
            
            
    def _watchdogTask(self):
        pass
            
    # disconnect device
    def disconnect(self):
        self.adapter.close()

def _test():
    return

if __name__ == "__main__":
    _test()
    

