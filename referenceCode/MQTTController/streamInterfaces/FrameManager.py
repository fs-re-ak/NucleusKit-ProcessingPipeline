


from streamInterfaces.proxies.OSCServerProxy import OSCServerProxy
from streamInterfaces.proxies.MQTTClientProxy import MQTTClientProxy
from streamInterfaces.proxies.UnixSocketProxy import UnixSocketProxy
from streamInterfaces.adapters.FrameAdapter import *

class FrameManager():

    def __init__(self, configuration, streamConfig, deviceFilter="*", showFrame=False, stubbed=False):

        if not stubbed:
            if configuration == "IPC":
                self.adapter = FrameAdapterOSC(showOSC=showFrame, stubbed=stubbed) #OSC works fine here
                self.proxy = UnixSocketProxy("CLIENT")
                self.proxy.attachAdapter(self.adapter)
            elif configuration == "OSC":
                self.adapter = FrameAdapterOSC(showOSC=showFrame, stubbed=stubbed)
                self.proxy = OSCServerProxy(filter=deviceFilter)
                self.proxy.attachCallback(self.adapter.msgCallback)
            elif configuration == "MQTT":
                
                broker_hostname = streamConfig.get("mqtt",{}).get("broker",{}).get("ip","localhost")
                port = streamConfig.get("mqtt",{}).get("broker",{}).get("port", 1883)
                topic = [streamConfig.get("streams",{}).get("MQTTFrameStream",{}).get("topic", None)]
                
                self.adapter = FrameAdapterMQTT(showMQTT=showFrame, stubbed=stubbed) #OSC works fine here
                self.proxy = MQTTClientProxy(broker_hostname=broker_hostname, port=port, in_routes=topic)
                self.proxy.attachCallback(self.adapter.msgCallback)
        else:
            self.proxy = None

        pass

    def attachCallback(self, callback):
        self.adapter.addCallback(callback)



