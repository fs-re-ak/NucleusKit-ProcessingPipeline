import time
import paho.mqtt.client as mqtt

class MQTTClientProxy:
    def __init__(self, broker_hostname="localhost", port=1883, in_routes=None, out_route=None, callback=None):
        
        
        print()
        print(f"[MQTTClientProxy] -----   MQTT frame stream   ----------")
        print(f"[MQTTClientProxy] MQTT frame stream")
        print(f"[MQTTClientProxy] host {broker_hostname}:{port}")
        print(f"[MQTTClientProxy] in_route {in_routes}")
        print(f"[MQTTClientProxy] --------------------------------------")
        print()
        
        
        
        self.broker_hostname = broker_hostname
        self.port = port
        self.in_routes = in_routes or []
        self.out_route = out_route
        self.callback = callback
        self.connected = False

        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self._on_connect
        self.mqttc.on_disconnect = self._on_disconnect
        self.mqttc.on_message = self._on_message

        # Let paho handle reconnect backoff
        self.mqttc.reconnect_delay_set(min_delay=1, max_delay=30)
        self.mqttc.connect_async(self.broker_hostname, self.port, 60)
        self.mqttc.loop_start()

    def attachCallback(self, callback):
        self.callback = callback

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = True
        for route in self.in_routes:
            client.subscribe(route)

    def _on_disconnect(self, client, userdata, reason_code, properties):
        self.connected = False

    def _on_message(self, client, userdata, msg):
        if self.callback:
            payload = msg.payload.decode("utf-8", errors="replace")
            self.callback(msg.topic, payload)

    def write(self, msg, qos=0, retain=False):
        if not self.connected:
            return False

        if self.out_route is not None:
            self.mqttc.publish(self.out_route, msg, qos=qos, retain=retain)

        return True

    def close(self):
        self.mqttc.disconnect()
        self.mqttc.loop_stop()
