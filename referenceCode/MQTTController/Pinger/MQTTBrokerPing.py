import os
import socket
from pythonosc import udp_client
from threading import Thread
from time import sleep

class MQTTBrokerPing():

    def __init__(self, ip=None, port=10339):
        self.port = port
        self.adapter = None

        if ip is None:
            ip = socket.gethostbyname(socket.gethostname())
            ip = ip.split('.')[:-1]
            ip.append("255")
            ip = ".".join(ip)
            print(f"Broadcast IP: {ip}")

        self.client = udp_client.SimpleUDPClient(ip, port)
        self.client._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self.pingThread = Thread(target=self.pingerTask)
        self.pingThread.daemon = True
        self.pingThread.start()

        pass


    def pingerTask(self):
        while True:
            self.client.send_message("/brokerPing", socket.gethostbyname(socket.gethostname()))
            sleep(1)



if __name__ == "__main__":
    pinger = MQTTBrokerPing()
    while True:
        sleep(1)