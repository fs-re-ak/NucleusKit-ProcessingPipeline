
from deviceController.DeviceControllerAdapter import DeviceControllerAdapter
from deviceController.proxies.UnixSocketProxy import UnixSocketProxy
from deviceController.proxies.OSCClientProxy import OSCClientProxy
from deviceController.proxies.MQTTClientProxy import MQTTClientProxy
from deviceController.proxies.MQTTBrokerPing import MQTTBrokerPing



class DeviceControllerManager():

    # single instance
    __instance = None

    # constructor that returns a reference
    def __new__(cls, configuration, streamConfig=None, stubbed=False):

        # if instance doesn't exist, yet (first call)
        if DeviceControllerManager.__instance is None:
            # define it
            DeviceControllerManager.__instance = object.__new__(cls)

            # set the value (template only)
            if configuration is not None:
                DeviceControllerManager.__instance._configure(configuration, streamConfig, stubbed)

        # return reference to instance
        return DeviceControllerManager.__instance

    def _configure(self, configuration, streamConfig, stubbed):

        self.stubbed = stubbed

        if not self.stubbed:

            if configuration=='IPC':
                self.proxy = UnixSocketProxy("CLIENT")
                self.adapter = DeviceControllerAdapter()
                self.proxy.attachAdapter(self.adapter)
                self.adapter.attachProxy(self.proxy)
            elif configuration=='OSC':
                self.proxy = OSCClientProxy()
                self.adapter = DeviceControllerAdapter()
                self.proxy.attachAdapter(self.adapter)
                self.adapter.attachProxy(self.proxy)
            elif configuration=='MQTT':
                
                if streamConfig.get("mqtt",{}).get("broker",{}).get("source","") == "Config":
                    self.proxy = MQTTClientProxy(host=streamConfig.get("mqtt",{}).get("broker",{}).get("ip","localhost"), 
                                                 port=streamConfig.get("mqtt",{}).get("broker",{}).get("port",1883),
                                                 out_route=streamConfig.get("mqtt",{}).get("controller",{}).get("topic",None))
                else:
                    self.pinger = MQTTBrokerPing()
                    self.proxy = MQTTClientProxy()
                self.adapter = DeviceControllerAdapter()
                self.proxy.attachAdapter(self.adapter)
                self.adapter.attachProxy(self.proxy)


            self.commandObserversCallback = []
            self.tagObserversCallback = []


    def sendStartRecording(self):

        if not self.stubbed:
            self.adapter.sendCommand("START_RECORDING")
        else:
            print("[DeviceControllerManager] stubbed, Start Recording")
        pass

    def sendStopRecording(self):
        if not self.stubbed:
            self.adapter.sendCommand("STOP_RECORDING")
        else:
            print("[DeviceControllerManager] stubbed, Stop Recording")
        pass

    def sendTag(self, tagId, values=[]):

        if not self.stubbed:
            self.adapter.sendTag(tagId, values)
        else:
            print(f"[DeviceControllerManager] stubbed, send tag {tagId}:{values}")
        pass