
import os
import json
from threading import Thread
from time import sleep, time
import random

from streamInterfaces.adapters.frameUtils import createFrame, assignRandomValues

class FrameAdapterOSC():

    def __init__(self, showOSC=False, stubbed=False):

        self.manager = None
        self.frame = {}
        self.callbacks = []
        self.stubbed = stubbed
        self.showOSC = showOSC

        periodicCallbackThread = Thread(target=self.periodicCallbackTask)
        periodicCallbackThread.daemon = True
        periodicCallbackThread.start()

        pass


    def periodicCallbackTask(self):

        while True:

            if self.stubbed:
                assignRandomValues(self.frame)

            for callback in self.callbacks:
                callback(self.frame)

            sleep(0.5)

    def addCallback(self, callback):
        self.callbacks.append(callback)


    def msgCallback(self, address, message):
        
        if self.showOSC:
            if "datatype" in address:  
                os.system('cls' if os.name == 'nt' else 'clear')  # Clears the terminal
            print(f"{address} {message}")
        
        parts = address.strip('/').split('/')

        # Skip device_id: parts[0] is 'vizia-nookii-1' or similar
        keys = parts[1:]  # ['status', 'is_recording'], etc.

        # Start from the root frame dict
        current = self.frame
        
        self.frame["device_id"] = parts[0]

        # Walk down the dict tree, creating subdicts if needed
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Assign the value at the final key
        leaf_key = keys[-1]

        # Convert value to bool if it's 0 or 1 and the key suggests it's boolean
        if isinstance(message, (int, float)) and message in (0, 1) and leaf_key in {
            "is_recording", "calibration_active", "headset_connected"
        }:
            print(message)
            value = bool(message)
        else:
            value = float(message) if isinstance(message, (int, float)) else message

        current[leaf_key] = value
        


class FrameAdapterMQTT():

    def __init__(self, showMQTT=False, stubbed=False):

        self.manager = None
        self.frame = {}
        self.callbacks = []
        self.stubbed = stubbed
        self.showMQTT = showMQTT

        pass


    def addCallback(self, callback):
        self.callbacks.append(callback)

    def msgCallback(self, topic, message):
        
        message = json.loads(message)
        
        if self.showMQTT:
            print(json.dumps(message,indent=2))
            
        for callback in self.callbacks:
            callback(message)
        

