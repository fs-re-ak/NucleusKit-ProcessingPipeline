import socket
import os

from threading import Thread

from time import sleep


class UnixSocketConfigs():
    # Set the path for the Unix socket
    SOCKET_PATH = '/ramdisk/frameSocket'


class UnixSocketProxy():

    def __init__(self, config):

        self.config = config
        self.adapter = None

        if config == "CLIENT":

            # Create the Unix socket client
            self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            # Connect to the server
            self.client.connect(UnixSocketConfigs.SOCKET_PATH)

            clientThread = Thread(target=self._receiveTaskClient, args=(self.client,))
            clientThread.daemon = True
            clientThread.start()

        elif config == "SERVER":

            self.serverConnexionThread = None
            self.serverClientConnections = []

            # remove the socket file if it already exists
            try:
                os.unlink(UnixSocketConfigs.SOCKET_PATH)
            except OSError:
                if os.path.exists(UnixSocketConfigs.SOCKET_PATH):
                    raise

            # Create the Unix socket server
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            # Bind the socket to the path
            self.server.bind(UnixSocketConfigs.SOCKET_PATH)

            self.serverConnexionThread = Thread(target=self._serverTask)
            self.serverConnexionThread.daemon = True
            self.serverConnexionThread.start()


        else:
            print("[UnixSocketProxy] Error, invalid config")
        pass


    def attachAdapter(self, adapter):
        self.adapter = adapter

    def _serverTask(self):

        while True:
            try:
                # Listen for incoming connections
                self.server.listen(1)

                # accept connections
                print('Server is listening for incoming connections...')
                connection, client_address = self.server.accept()

                self.serverClientConnections.append(connection)

                clientThread = Thread(target=self._receiveTaskServer, args=(connection,))
                clientThread.daemon = True
                clientThread.start()
            except:
                break

    def send(self, message):

        if self.config == "CLIENT":
            self.client.sendall(message.encode())

        elif self.config == "SERVER":
            for connection in self.serverClientConnections:
                print(f"send to connection {connection}")
                connection.sendall(message.encode())
        pass


    def _receiveTaskClient(self, params):
        connection = params

        try:
            # receive data from the client
            while True:
                data = connection.recv(1024)
                if not data:
                    break
                if self.adapter is not None:
                    print(f"proxy rcv msg: {data.decode}")
                    self.adapter.msgCallback(data.decode())
                else:
                    print('Received data:', data.decode())

        except Exception as e:
            print(f'[UnixSocketProxy] Connection receive failed: {e}')
        pass

    def _receiveTaskServer(self, params):
        connection = params

        try:
            #print('Connection from', str(connection).split(", ")[0][-4:])

            # receive data from the client
            while True:
                data = connection.recv(1024)
                if not data:
                    break

                if self.adapter is not None:
                    self.adapter.msgCallback(data.decode())
                else:
                    print('Server received data:', data.decode())

        except Exception as e:
            print(f'[UnixSocketProxy] Connection receive failed: {e}')
            self.serverClientConnections.remove(connection)
        pass

    def disconnect(self):
        if self.config == "CLIENT":
            self.client.close()

        elif self.config == "SERVER":
            for connection in self.serverClientConnections:
                # close the connection
                connection.close()

            # remove the socket file
            os.unlink(UnixSocketConfigs.SOCKET_PATH)
        pass



if __name__ == "__main__":
    server = UnixSocketProxy("SERVER")
    clientA = UnixSocketProxy("CLIENT")
    clientB = UnixSocketProxy("CLIENT")

    server.send("Server to clients")

    clientA.send("ClientA to server")
    clientB.send("ClientB to server")

    sleep(5)
    server.disconnect()


