
import json


class DeviceControllerAdapter():


    def __init__(self):
        self.proxy = None
        pass


    def attachProxy(self,proxy):
        self.proxy = proxy

    def sendCommand(self,command):

        message = {"Type":"COMMAND","Content":{"Code":command}}
        self.proxy.write(json.dumps(message))

        pass

    def sendTag(self,tag,values):

        message = {"Type":"TAG","Content":{"Code":tag,"Values":str(values)}}
        self.proxy.write(json.dumps(message))

        pass
