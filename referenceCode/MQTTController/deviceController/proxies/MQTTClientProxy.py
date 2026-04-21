
import os
import socket
from pythonosc import udp_client

import paho.mqtt.client as mqtt
from threading import Thread
from time import sleep


class MQTTClientProxy():

    def __init__(self, host=None, port=1883, routes = [], out_route=None):

        print()
        print(f"[MQTTClientProxy] -----   MQTT controller (from remote)   ----------")
        print(f"[MQTTClientProxy] MQTT frame stream")
        print(f"[MQTTClientProxy] host {host}:{port}")
        print(f"[MQTTClientProxy] out_route {out_route}")
        print(f"[MQTTClientProxy] --------------------------------------------------")
        print()


        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self._on_connect
        self.mqttc.on_disconnect = self._on_disconnect
        self.mqttc.on_message = self._on_message
        self.routes = routes
        self.out_route = out_route
        
        if host is None:
            self.host = socket.gethostbyname(socket.gethostname())
        else:
            self.host = host
        self.port = port

        # non blocking, handles reconnect
        self.mqttc.loop_start()

        self.thread = Thread(target=self._connectTask)
        self.thread.daemon = True
        self.thread.start()

    def attachAdapter(self, adapter):
        self.adapter = adapter

    def _connectTask(self):
        self.connected = False

        while not self.connected:
            try:
                print(f"connecting host: {self.host}:{self.port}")
                self.mqttc.connect(self.host, self.port, 60)
                self.connected = True
            except Exception as e:
                print(f"MQTT - connection failed {e}, retrying")
                pass


    # The callback for when the client receives a CONNACK response from the server.
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")

        for route in self.routes:
            client.subscribe(route)

    def _on_disconnect(self, Client, userdata, flags, reason_code, properties):
        print(f"Disconnected with result code {reason_code}")

        self.thread = Thread(target=self._connectTask)
        self.thread.daemon = True
        self.thread.start()

    # The callback for when a PUBLISH message is received from the server.
    def _on_message(self, client, userdata, msg):
        if self.adapter is not None:
            self.adapter.decode(msg.topic + " " + str(msg.payload))

    def write(self, msg):
        if self.connected:
            print(msg)
            try:
                if self.out_route is not None:
                    self.mqttc.publish(self.out_route, msg)
            except Exception as e:
                print(f"[MQTTClientProxy.py] Error sending {e}")
                pass
        else:
            print("[MQTTClientProxy.py] Trying to send, when not connected")

    def close(self):
        self.mqttc.disconnect()




if __name__ == "__main__":
    class Adapter():
        def decode(self, msg):
            print(msg)

    proxy = MQTTClientProxy(host="192.168.50.202")
    adapter = Adapter()

    proxy.attachAdapter(adapter)

    sleep(3)
    proxy.write("control/","Hello")

    sleep(200)