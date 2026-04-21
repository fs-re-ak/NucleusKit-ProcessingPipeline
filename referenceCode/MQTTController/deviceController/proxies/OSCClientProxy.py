
import socket
from pythonosc import udp_client

class OSCClientProxy(object):

    port = None

    def __init__(self, ip=socket.gethostbyname(socket.gethostname()), port=10338):
        self.port = port
        self.adapter = None

        if ip is None:
            ip = socket.gethostbyname(socket.gethostname())
            ip = ip.split('.')[:-1]
            ip.append("255")
            ip = ".".join(ip)
            print(f"Own IP: {ip}")

        self.client = udp_client.SimpleUDPClient(ip, port)
        self.client._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        pass

    def attachAdapter(self, adapter):
        self.adapter = adapter

    def write(self, msg, route="/control"):
        self.client.send_message(route, msg)

    def close(self):
        pass
