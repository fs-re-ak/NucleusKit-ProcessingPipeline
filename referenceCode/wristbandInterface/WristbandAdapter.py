"""
WristbandAdapter (Template)
    - links the headset and the output destination

Roles:
    - init output file and route information (add inferred time)
    - delay recording start
    
Design:
    - provides a factory to initialise object 
    - Muse is always streaming, choice is made whether data makes it to output file
    
TODO: 
    - add fake muse
    
Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Spring 2018
"""
from abc import ABC, abstractmethod
from tools.csvFile.CSVFileInterface import CSVFileInterface
from time import time, sleep
import pickle
import numpy as np
import hickle as hkl
 
"""
AdapterFactory, arbiter the creation of the adapter, based on requested configuration.

Public:
    - factory(), static method to get a reference to requested object
    - thishelp(), static method that shows (console) list of available configurations
"""
class WristbandAdapterFactory(object):

    """
        Name: factory
        Descr: return a reference to requested object
        Parameters:
            configuration: string of value: {"MUSE"}
            manager: reference to the manager
            
        Return:
            reference to requested adapter, None otherwise
    """
    def factory(configuration, manager):
        
        if configuration == "SHIMMER-IO": return AdapterShimmerIO(manager)
        elif configuration == "SHIMMER-SERIAL": return AdapterShimmerSerial(manager)
        else: 
            print('[WristbandAdapter.py] invalid factory configuration')
            return None
        
    factory = staticmethod(factory)

    def thishelp():
        print("Available configurations:")
        print("- SHIMMER_GSR")
        
    thishelp = staticmethod(thishelp) 
 
 
 
"""
AdapterBase, abstract class that defines the base of all Adapters

Public:
    - __init__(), base constructor (require reference to manager)
    - activate(), to flip an object to active state
    - deactivate(), to flip an object to non-active state
    - attachProxy(), to attach proxy to adapter
    - connect(), connect to device
    - startRecording(), to begin recording
    - stopRecording(), to stop recording
    - disconnect(), to disconnect from device
"""
class WristbandAdapterBase(ABC):
    
    proxy = None
    manager = None
    
    active = False

    outFrameStreamHandles = []
    outRawDataHandles = []
 
    """
        Name: __init__
        Descr: base constructor
        Parameters:
            managerHandle: reference to manager
    """
    def __init__(self, managerHandle):
        self.manager = managerHandle
        super().__init__()
    
    """
        Name: activate
        Descr: make object active, after successfully built
    """
    def activate(self):
        self.active = True
        pass
        
    """
        Name: deactivate
        Descr: make object inactive, without destroying it
    """
    def deactivate(self):
        self.active = False
        pass
        
    """
        Name: attachAdapter
        Descr: copy reference to proxy, should be called before object is made active
        Parameters:
            proxyHandle: reference to proxy
            
    """
    def attachProxy(self, proxyHandle):
        self.proxy = proxyHandle
        pass


    def attachOutFrameStream(self, outFrameStreamHandle):
        self.outFrameStreamHandles.append(outFrameStreamHandle)

    def attachOutRawDataStream(self, outRawDataHandle):
        self.outRawDataHandles.append(outRawDataHandle)

    @abstractmethod
    def connect(self):
        pass
        
    @abstractmethod    
    def startRecording(self, startDelay=0):
        pass

    @abstractmethod
    def postProcess(self):
        pass
        
    @abstractmethod    
    def stopRecording(self):
        pass
        
    @abstractmethod    
    def disconnect(self):
        pass


"""
AdapterStub, instantiable adapter for a stubbed headset

Public:
    - connect(), attempt at pairing with the device
    - startStreaming(), begin streaming data
    - startStreaming(), stop streaming data
    - disconnect(), disconnect from device
"""
class AdapterStub(WristbandAdapterBase):

    def __init__(self, managerHandle):
        super().__init__(managerHandle)

    """
        Name: connect
        Descr: shows a message in console and ask proxy to do the same
    """
    def connect(self):
        if self.active:
            return self.proxy.connect()
        return False
         
    def startRecording(self, startDelay=0):
        pass
        
    def stopRecording(self):
        pass

    def postProcess(self):
        pass
        
    def disconnect(self):
        if self.active:
            self.proxy.close()
        pass


"""
AdapterStub, instantiable adapter for a stubbed headset

Public:
    - connect(), attempt at pairing with the device
    - startStreaming(), begin streaming data
    - startStreaming(), stop streaming data
    - disconnect(), disconnect from device
"""
class AdapterShimmerIO(WristbandAdapterBase):

    outputFilename = None
    recording = False

    def __init__(self, managerHandle):
        super().__init__(managerHandle)

    """
        Name: connect
        Descr: shows a message in console and ask proxy to do the same
    """
    def connect(self):
        if self.active:
            _, status = self.proxy.onEvent("STATUS", **{})
            if "CONNECTED" in status:
                print('[WristbandAdapter.py] reported connected')
                return True
            return self.proxy.connect()
        return False
        
    # setup output file for recording
    def setupRecorder(self, sessionPath):
        self.outputFilename = sessionPath + "/wbData.hkl"
        pass
        
    # call to start recording
    def startRecording(self, startDelay=0):
        
        # wait for delay before starting
        if startDelay>0:
            print(('[WristbandRecorderAdapter.py] startDelay: %i') % startDelay)
            sleep(startDelay)

        self.recording = True
        self.proxy.onEvent("RECORDING", **{"state":True})
        pass

    def postProcess(self):
        print('[WristbandRecorderAdapter.py] Post-processing')
        res = self.proxy.onEvent("POST-PROCESS", **{})
        if res is not None:
            proxyId, *_ = res
            if proxyId == "shimmer-3":
                _, idx, lst = res
                lst = lst[:idx,:]
                #pickle.dump(lst, self.outputFile)
                hkl.dump(lst, self.outputFilename, mode='w', compression='gzip')
                #self.outputFile.close()
        pass
        
    def stopRecording(self):
        # stop and close file
        self.recording = False
        self.proxy.onEvent("RECORDING", **{"state": False})
        pass

    def disconnect(self):
        if self.active:
            self.proxy.close()
        pass


"""
AdapterStub, instantiable adapter for a stubbed headset

Public:
    - connect(), attempt at pairing with the device
    - startStreaming(), begin streaming data
    - startStreaming(), stop streaming data
    - disconnect(), disconnect from device
"""


class AdapterShimmerSerial(WristbandAdapterBase):

    def __init__(self, managerHandle):
        super().__init__(managerHandle)

    def connect(self):
        return self.proxy.connect()
        
    # setup output file for recording
    def setupRecorder(self, sessionPath):
        pass

    # call to start recording
    def startRecording(self, startDelay=0):

        # wait for delay before starting
        if startDelay > 0:
            print(('[WristbandRecorderAdapter.py] startDelay: %i') % startDelay)
            sleep(startDelay)

        self.proxy.onEvent("RECORDING", **{"state": True})
        pass

    def postProcess(self):
        pass

    def stopRecording(self):
        # stop and close file
        self.proxy.onEvent("RECORDING", **{"state": False})
        pass

    def recvData(self, data):
        for handle in self.outRawDataHandles:
            handle.setData([data])

    def disconnect(self):
        if self.active:
            self.proxy.close()
        pass

